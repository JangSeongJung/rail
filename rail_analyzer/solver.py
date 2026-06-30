"""3D 프레임 직접강성법 솔버 (요소당 12 DOF).

강성행렬 K는 이동하중 위치와 무관하게 일정하므로, 자유도 분할 후
자유-자유 강성행렬을 한 번만 역행렬로 분해해두고 위치별 하중벡터에 대해
빠르게 반복 해석한다. 각 위치에서 변위/반력/부재 단면력을 복원한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import StructuralModel


def _local_stiffness(E, G, A, Iy, Iz, J, L) -> np.ndarray:
    """국부좌표 12x12 빔 강성행렬.

    DOF 순서: [u1,v1,w1,rx1,ry1,rz1, u2,v2,w2,rx2,ry2,rz2]
    u=축(x), v=국부y, w=국부z, rx=비틀림, ry=y축휨, rz=z축휨.
    """
    k = np.zeros((12, 12))
    EA_L = E * A / L
    GJ_L = G * J / L

    # 축력
    k[0, 0] = k[6, 6] = EA_L
    k[0, 6] = k[6, 0] = -EA_L
    # 비틀림
    k[3, 3] = k[9, 9] = GJ_L
    k[3, 9] = k[9, 3] = -GJ_L

    # x-y 평면 휨 (v, rz) : Iz
    a = 12 * E * Iz / L ** 3
    b = 6 * E * Iz / L ** 2
    c = 4 * E * Iz / L
    d = 2 * E * Iz / L
    k[1, 1] = a;  k[1, 5] = b;  k[1, 7] = -a;  k[1, 11] = b
    k[5, 1] = b;  k[5, 5] = c;  k[5, 7] = -b;  k[5, 11] = d
    k[7, 1] = -a; k[7, 5] = -b; k[7, 7] = a;   k[7, 11] = -b
    k[11, 1] = b; k[11, 5] = d; k[11, 7] = -b; k[11, 11] = c

    # x-z 평면 휨 (w, ry) : Iy
    a2 = 12 * E * Iy / L ** 3
    b2 = 6 * E * Iy / L ** 2
    c2 = 4 * E * Iy / L
    d2 = 2 * E * Iy / L
    k[2, 2] = a2;  k[2, 4] = -b2; k[2, 8] = -a2; k[2, 10] = -b2
    k[4, 2] = -b2; k[4, 4] = c2;  k[4, 8] = b2;  k[4, 10] = d2
    k[8, 2] = -a2; k[8, 4] = b2;  k[8, 8] = a2;  k[8, 10] = b2
    k[10, 2] = -b2; k[10, 4] = d2; k[10, 8] = b2; k[10, 10] = c2
    return k


def _transform(dx: np.ndarray) -> np.ndarray:
    """요소 방향벡터 → 12x12 변환행렬 T (국부 = T · 전역)."""
    L = np.linalg.norm(dx)
    e1 = dx / L
    # 부재가 거의 연직이면 기준벡터를 전역 Y로
    if abs(e1[2]) > 0.999:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(ref, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R = np.vstack([e1, e2, e3])     # 3x3, 행 = 국부축
    T = np.zeros((12, 12))
    for blk in range(4):
        T[blk * 3:blk * 3 + 3, blk * 3:blk * 3 + 3] = R
    return T


@dataclass
class ElementData:
    nodes: tuple[int, int]
    L: float
    T: np.ndarray            # 12x12
    k_local: np.ndarray      # 12x12
    dof_index: np.ndarray    # (12,) 전역 DOF 인덱스


class FrameSolver:
    """3D 프레임 정적 솔버 (강성 사전분해 후 다중 하중 반복)."""

    def __init__(self, model: StructuralModel):
        self.model = model
        self.n_dof = model.n_dof
        s = model.section
        self.elements: list[ElementData] = []

        K = np.zeros((self.n_dof, self.n_dof))
        for (ni, nj) in model.elements:
            dx = model.nodes[nj] - model.nodes[ni]
            L = float(np.linalg.norm(dx))
            T = _transform(dx)
            kloc = _local_stiffness(s.E, s.G, s.A, s.Iy, s.Iz, s.J, L)
            kglob = T.T @ kloc @ T
            dof = np.r_[6 * ni:6 * ni + 6, 6 * nj:6 * nj + 6]
            K[np.ix_(dof, dof)] += kglob
            self.elements.append(ElementData((int(ni), int(nj)), L, T, kloc, dof))

        self.K = K

        # 구속/자유 DOF 분할
        fixed = np.zeros(self.n_dof, dtype=bool)
        for node, dofs in model.supports.items():
            for k6, con in enumerate(dofs):
                if con:
                    fixed[6 * node + k6] = True
        self.fixed = np.where(fixed)[0]
        self.free = np.where(~fixed)[0]

        Kff = K[np.ix_(self.free, self.free)]
        # 안정성 확인 후 역행렬 사전계산 (위치별 반복용)
        try:
            self.Kff_inv = np.linalg.inv(Kff)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "강성행렬이 특이행렬입니다. 지지조건이 부족해 구조가 불안정합니다."
            ) from exc
        self.Kfr = K[np.ix_(self.fixed, self.free)]

    def solve(self, F: np.ndarray) -> "SolveResult":
        """전역 하중벡터 F (n_dof,)에 대한 해석.

        Returns:
            SolveResult(변위, 반력, 요소 단면력)
        """
        U = np.zeros(self.n_dof)
        Ff = F[self.free]
        U[self.free] = self.Kff_inv @ Ff

        # 반력 = K[fixed,:] U - F[fixed]
        reactions_all = self.K[self.fixed, :] @ U - F[self.fixed]
        reactions = {int(dof): float(r) for dof, r in zip(self.fixed, reactions_all)}

        # 요소 단면력(국부): f_local = k_local · T · u_elem
        elem_forces = np.zeros((len(self.elements), 12))
        for e, ed in enumerate(self.elements):
            u_e = U[ed.dof_index]
            elem_forces[e] = ed.k_local @ (ed.T @ u_e)

        return SolveResult(U=U, reactions=reactions, elem_forces=elem_forces)


@dataclass
class SolveResult:
    U: np.ndarray                       # (n_dof,) 전역 변위
    reactions: dict[int, float]         # 구속 DOF -> 반력
    elem_forces: np.ndarray             # (E, 12) 요소 국부 단면력
