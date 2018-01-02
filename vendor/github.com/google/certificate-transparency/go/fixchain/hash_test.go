package fixchain

import (
	"testing"
)

func TestHashBag(t *testing.T) {
	hashBagTests := []struct {
		certList1 []string
		certList2 []string
		expEqual  bool
		errMsg    string
	}{
		{
			[]string{googleLeaf},
			[]string{thawteIntermediate},
			false,
			"hash match between Bags containing different certs",
		},
		{
			[]string{googleLeaf, thawteIntermediate},
			[]string{thawteIntermediate, googleLeaf},
			true,
			"hash mismatch between Bags containing the same certs",
		},
		{
			[]string{googleLeaf, thawteIntermediate},
			[]string{thawteIntermediate, googleLeaf, thawteIntermediate},
			false,
			"hash match between Bags containing the same certs, but one with duplicates",
		},
	}

	for i, test := range hashBagTests {
		certList1 := extractTestChain(t, i, test.certList1)
		certList2 := extractTestChain(t, i, test.certList2)
		if (hashBag(certList1) == hashBag(certList2)) != test.expEqual {
			t.Errorf("#%d: %s", i, test.errMsg)
		}
	}
}
