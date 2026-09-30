from core.majority import MajorityClass


def test_predicts_most_frequent_label_seen_so_far():
    model = MajorityClass()
    assert model.predict_one([]) == 0  # tie before any data
    for label in (1, 1, 0):
        model.learn_one(["x"], label)
    assert model.predict_one(["anything"]) == 1
    model.learn_one([], 0)
    model.learn_one([], 0)
    assert model.predict_one([]) == 0
