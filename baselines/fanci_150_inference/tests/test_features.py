import numpy as np

from domain_inference.models.fanci_features import FEATURE_NAMES, OfficialFANCI45Extractor


EXTRACTOR = OfficialFANCI45Extractor(
    valid_tlds={".de", ".hm", ".uk"},
    valid_public_suffixes={".de", ".hm", ".co.uk", ".uk"},
)


def test_feature_order_and_official_source_expectations():
    examples = {
        "rwth": "itsec.rwth-aachen.de",
        "digits": "123huhu384gj8nge8.co.uk",
        "subdomains": "1.2.3.4.5.6.7.8.9.123huhu384gj8nge8.co.uk",
        "vowel_one": "eeeeeeeeeeeeeee",
        "vowel_zero": "ggggg123ggghjkl12323.de",
        "mhm": "m.hm",
        "one_digit": "1",
        "one_letter": "a",
        "only_tld": "de",
        "double_tld": "de.de",
        "triple_tld": "de.de.de",
        "ip_addr": "hu.10.13.37.19.rwth-aachen.de",
        "wwwdot": "wegonweg.www.egioegn.de",
        "wwwtrailingdot": "www.egioegn.de",
        "prefix_repeat": "rwth-aachen.derwth-aachen.de",
        "ngrameasy": "abcdef.de",
        "underscores": "_tcp.ab_cdef1.de",
    }
    feature = {name: EXTRACTOR.extract(domain) for name, domain in examples.items()}
    assert len(FEATURE_NAMES) == 45
    assert all(vector.shape == (45,) for vector in feature.values())
    assert all(np.isfinite(vector).all() for vector in feature.values())

    expectations = [
        (feature["rwth"][0], len(examples["rwth"])),
        (feature["digits"][0], len(examples["digits"])),
        (feature["rwth"][1:5], [0, 1, 0, 0]),
        (feature["digits"][1:5], [1, 0, 0, 0]),
        (feature["subdomains"][1:5], [0, 0, 0, 1]),
        (feature["rwth"][5], 5 / 15),
        (feature["digits"][5], 3 / 9),
        (feature["subdomains"][5], 3 / 9),
        (feature["vowel_one"][5], 1),
        (feature["vowel_zero"][5], 0),
        (feature["digits"][6], 8 / 17),
        (feature["rwth"][7], 0),
        (feature["one_letter"][7], 0),
        (feature["ip_addr"][7], 1),
        (feature["one_digit"][7], 0),
        (feature["mhm"][7], 0),
        (feature["ip_addr"][8], 1),
        (feature["one_digit"][8], 1),
        (feature["one_letter"][8], 0),
        (feature["rwth"][8], 0),
        (feature["rwth"][9], 1),
        (feature["one_digit"][9], 0),
        (feature["only_tld"][9], 0),
        (feature["vowel_zero"][9], 1),
        (feature["vowel_one"][9], 0),
        (feature["rwth"][10], 0),
        (feature["one_letter"][10], 1),
        (feature["one_digit"][10], 1),
        (feature["vowel_zero"][10], 0),
        (feature["vowel_one"][10], 0),
        (feature["subdomains"][10], 1),
        (feature["subdomains"][11], 0),
        (feature["wwwdot"][11], 1),
        (feature["wwwtrailingdot"][11], 1),
        (feature["rwth"][11], 0),
        (feature["rwth"][12], 8),
        (feature["subdomains"][12], 26 / 10),
        (feature["rwth"][13], 0),
        (feature["subdomains"][13], 0),
        (feature["double_tld"][13], 0),
        (feature["prefix_repeat"][13], 1),
        (feature["one_letter"][14], 1),
        (feature["one_digit"][14], 1),
        (feature["rwth"][14], 11 / 16),
        (feature["triple_tld"][14], 1 / 2),
        (feature["rwth"][15], 0),
        (feature["subdomains"][15], 1),
        (feature["one_digit"][15], 1),
        (feature["double_tld"][15], 0),
        (feature["ip_addr"][15], 1),
        (feature["prefix_repeat"][15], 0),
        (feature["rwth"][16], 0),
        (feature["subdomains"][16], 0),
        (feature["one_digit"][16], 0),
        (feature["triple_tld"][16], 1),
        (feature["prefix_repeat"][16], 0),
        (feature["ngrameasy"][38], 1),
        (feature["subdomains"][38], 9 / 10),
        (feature["rwth"][38], 0),
        (feature["rwth"][39], 0),
        (feature["subdomains"][39], 0),
        (feature["prefix_repeat"][39], 0),
        (feature["underscores"][39], 2 / 12),
        (feature["rwth"][40], 11),
        (feature["vowel_one"][40], 1),
        (feature["ngrameasy"][40], 6),
        (feature["one_digit"][40], 1),
        (feature["one_letter"][40], 1),
        (feature["vowel_one"][42], 1),
        (feature["one_letter"][42], 0),
        (feature["one_digit"][42], 0),
        (feature["rwth"][42], 5 / 11),
        (feature["one_digit"][43], 0),
        (feature["vowel_one"][43], 0),
        (feature["rwth"][43], 8 / 16),
        (feature["underscores"][43], 5 / 12),
        (feature["one_digit"][44], 0),
        (feature["vowel_one"][44], 0),
        (feature["rwth"][44], 0),
        (feature["underscores"][44], 0),
        (feature["digits"][44], 6 / 17),
    ]
    for actual, expected in expectations:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    for name in ("mhm", "one_digit", "one_letter"):
        np.testing.assert_allclose(feature[name][24:38], [-1] * 14, rtol=0, atol=0)
    for start in (17, 24, 31):
        np.testing.assert_allclose(
            feature["ngrameasy"][start:start + 7], [0, 1, 1, 1, 1, 1, 1], rtol=0, atol=0
        )
    np.testing.assert_allclose(feature["rwth"][41], 3.375, rtol=0, atol=0.1)
    np.testing.assert_allclose(feature["vowel_one"][41], 0, rtol=0, atol=0.1)
    np.testing.assert_allclose(feature["digits"][41], 3.33718, rtol=0, atol=0.05)


def test_batch_matches_individual_extraction():
    domains = ["google.com", "6301un092rsh.org", "www.example.de"]
    batch = EXTRACTOR.extract_many(domains, n_jobs=1)
    expected = np.vstack([EXTRACTOR.extract(domain) for domain in domains])
    np.testing.assert_array_equal(batch, expected)
