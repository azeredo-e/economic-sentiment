from llm_econ_text.data.thesis_epu import thesis_epu_flag


def test_needs_both_p_and_u():
    assert thesis_epu_flag("the government may delay the decision")
    assert not thesis_epu_flag("the government opened a factory")  # P only
    assert not thesis_epu_flag("sales may rise next quarter")  # U only


def test_multiword_p_terms_never_match():
    assert not thesis_epu_flag("the central bank may delay")  # 'central bank' is multi-word
