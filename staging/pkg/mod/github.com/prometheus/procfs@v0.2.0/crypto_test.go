// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestFS_Crypto(t *testing.T) {
	fs := getProcFixtures(t)
	crypto, err := fs.Crypto()

	if err != nil {
		t.Fatalf("parsing of reference-file failed entirely: %s", err)
	}

	refs := []Crypto{
		{
			Name:        "ccm(aes)",
			Driver:      "ccm_base(ctr(aes-aesni),cbcmac(aes-aesni))",
			Module:      "ccm",
			Priority:    newint64(300),
			Refcnt:      newint64(4),
			Selftest:    "passed",
			Internal:    "no",
			Type:        "aead",
			Async:       false,
			Blocksize:   newuint64(1),
			Ivsize:      newuint64(16),
			Maxauthsize: newuint64(16),
			Geniv:       "<none>",
		},
		{
			Name:       "cbcmac(aes)",
			Driver:     "cbcmac(aes-aesni)",
			Module:     "ccm",
			Priority:   newint64(300),
			Refcnt:     newint64(7),
			Selftest:   "passed",
			Internal:   "no",
			Type:       "shash",
			Blocksize:  newuint64(1),
			Digestsize: newuint64(16),
		},
		{
			Name:     "ecdh",
			Driver:   "ecdh-generic",
			Module:   "ecdh_generic",
			Priority: newint64(100),
			Refcnt:   newint64(1),
			Selftest: "passed",
			Internal: "no",
			Type:     "kpp",
			Async:    true,
		},
		{
			Name:       "ecb(arc4)",
			Driver:     "ecb(arc4)-generic",
			Module:     "arc4",
			Priority:   newint64(100),
			Refcnt:     newint64(1),
			Selftest:   "passed",
			Internal:   "no",
			Type:       "skcipher",
			Async:      false,
			Blocksize:  newuint64(1),
			MinKeysize: newuint64(1),
			MaxKeysize: newuint64(256),
			Ivsize:     newuint64(0),
			Chunksize:  newuint64(1),
			Walksize:   newuint64(1),
		},
		{
			Name:       "arc4",
			Driver:     "arc4-generic",
			Module:     "arc4",
			Priority:   newint64(0),
			Refcnt:     newint64(3),
			Selftest:   "passed",
			Internal:   "no",
			Type:       "cipher",
			Blocksize:  newuint64(1),
			MinKeysize: newuint64(1),
			MaxKeysize: newuint64(256),
		},
		{
			Name:       "crct10dif",
			Driver:     "crct10dif-pclmul",
			Module:     "crct10dif_pclmul",
			Priority:   newint64(200),
			Refcnt:     newint64(2),
			Selftest:   "passed",
			Internal:   "no",
			Type:       "shash",
			Blocksize:  newuint64(1),
			Digestsize: newuint64(2),
		},
	}

	if want, have := len(refs), len(crypto); want > have {
		t.Errorf("want at least %d parsed crypto-entries, have %d", want, have)
	}
	for index, ref := range refs {
		want, got := ref, crypto[index]
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("unexpected crypto entry (-want +got):\n%s", diff)
		}
	}
}

func newint64(i int64) *int64 {
	return &i
}

func newuint64(i uint64) *uint64 {
	return &i
}
