"""레일 이동하중 검토 - GUI 실행 진입점.

실행:
    python app.py
DXF를 고르고 사람/트롤리 무게·거리(줄 길이)·최고속도·마력을 입력하면
동·정역학 3케이스 하중 리포트(HTML)와 MGT를 생성한다.
"""
from rail_analyzer.report_gui import main

if __name__ == "__main__":
    main()
