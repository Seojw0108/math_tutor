def setup_korean_font():
    """설치된 한글 글꼴 하나를 골라 matplotlib 에 설정 (Windows: 맑은 고딕, Linux: 나눔/Noto)."""
    import logging

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "UnDotum"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt
