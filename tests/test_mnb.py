import math
from collections import Counter

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from core.metrics import prequential
from core.mnb import AccumulativeMNB, AgeingMNB


def synthetic_corpus(n_docs: int, seed: int = 0, vocab_size: int = 300):
    """Documents whose word distribution depends on the class, so NB has signal."""
    rng = np.random.default_rng(seed)
    vocab = np.array([f"w{i}" for i in range(vocab_size)])
    word_probs = rng.dirichlet(np.full(vocab_size, 0.3), size=2)
    labels = rng.integers(0, 2, size=n_docs)
    docs = [list(rng.choice(vocab, size=rng.integers(3, 15), p=word_probs[y])) for y in labels]
    return docs, labels.tolist()


def naive_ageing_scores(model: AgeingMNB, tokens, t):
    """Eq. 1-2 evaluated literally from the stored counts, with no caching."""
    counts = Counter(w for w in tokens if w in model.vocab)
    av = model.alpha * len(model.vocab)
    scores = []
    for c in (0, 1):
        words = model.words[c]
        s = math.log(model.class_count[c]) - model.lam * (t - model.class_tlo[c])
        denom = sum(n * math.exp(-model.lam * (t - tlo)) for n, tlo in words.values()) + av
        for w, k in counts.items():
            n, tlo = words.get(w, (0, t))
            s += k * (math.log(n * math.exp(-model.lam * (t - tlo)) + model.alpha) - math.log(denom))
        scores.append(s)
    return scores


def test_accumulative_matches_sklearn_multinomial_nb():
    docs, labels = synthetic_corpus(3000)
    train_docs, train_y, test_docs = docs[:2500], labels[:2500], docs[2500:]

    ours = AccumulativeMNB(alpha=1.0)
    for tokens, y in zip(train_docs, train_y):
        ours.learn_one(tokens, y)

    vec = CountVectorizer(analyzer=lambda d: d)
    X = vec.fit_transform(train_docs)
    ref = MultinomialNB(alpha=1.0)
    # Stream the training set through partial_fit in chunks, as a streaming learner would.
    for start in range(0, X.shape[0], 500):
        ref.partial_fit(X[start:start + 500], train_y[start:start + 500], classes=[0, 1])

    X_test = vec.transform(test_docs)
    ref_scores = ref.predict_joint_log_proba(X_test)
    our_scores = np.array([ours.log_scores(d) for d in test_docs])
    np.testing.assert_allclose(our_scores[:, 1] - our_scores[:, 0],
                               ref_scores[:, 1] - ref_scores[:, 0], rtol=1e-9, atol=1e-9)
    assert [ours.predict_one(d) for d in test_docs] == ref.predict(X_test).tolist()


def test_ageing_with_zero_lambda_equals_accumulative_prequentially():
    docs, labels = synthetic_corpus(2000, seed=1)
    times = np.cumsum(np.random.default_rng(1).exponential(0.01, size=len(docs)))
    stream = list(zip(docs, labels, times))

    acc = prequential(AccumulativeMNB(), stream)
    ageing = prequential(AgeingMNB(lam=0.0), stream)
    assert np.array_equal(acc.y_pred, ageing.y_pred)


@pytest.mark.parametrize("lam", [0.05, 0.2, 1.0])
def test_ageing_cached_normaliser_matches_literal_equations(lam):
    # Unit time steps with lam up to 1 force several t_ref rebases along the way.
    docs, labels = synthetic_corpus(400, seed=2)
    model = AgeingMNB(lam=lam)
    probes = synthetic_corpus(20, seed=3)[0]
    for i, (tokens, y) in enumerate(zip(docs, labels)):
        t = float(i) + 0.5 * (i % 3)  # irregular but non-decreasing clock
        if i > 10 and i % 50 == 0:
            for probe in probes:
                np.testing.assert_allclose(model.log_scores(probe, t),
                                           naive_ageing_scores(model, probe, t), rtol=1e-9)
        model.learn_one(tokens, y, t)


def test_set_lambda_rebuilds_normaliser_exactly():
    docs, labels = synthetic_corpus(300, seed=4)
    model = AgeingMNB(lam=0.1)
    for i, (tokens, y) in enumerate(zip(docs, labels)):
        model.learn_one(tokens, y, float(i))
    model.set_lambda(0.6)
    probe, t = docs[0], 305.0
    np.testing.assert_allclose(model.log_scores(probe, t), naive_ageing_scores(model, probe, t),
                               rtol=1e-9)


def test_ageing_forgets_a_class_that_stops_appearing():
    # Same words in both classes; only recency differs.
    model = AgeingMNB(lam=0.2)
    for i in range(200):
        model.learn_one(["good", "day"], 1, t=float(i))  # class 1 dominates early...
    for i in range(20):
        model.learn_one(["good", "day"], 0, t=200.0 + i)  # ...then only class 0 appears
    assert model.predict_one(["good", "day"], t=220.0) == 0

    acc = AccumulativeMNB()
    for i in range(200):
        acc.learn_one(["good", "day"], 1)
    for i in range(20):
        acc.learn_one(["good", "day"], 0)
    assert acc.predict_one(["good", "day"]) == 1  # the accumulative model does not forget


def test_unknown_words_are_ignored():
    model = AgeingMNB(lam=0.1)
    model.learn_one(["a"], 0, 0.0)
    model.learn_one(["b"], 1, 1.0)
    assert model.log_scores(["a", "zzz"], 2.0) == model.log_scores(["a"], 2.0)
