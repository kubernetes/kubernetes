// Copyright 2016 The Linux Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package identity

import (
	_ "crypto/sha256" // required to install sha256 digest support
	"reflect"
	"testing"

	"github.com/opencontainers/go-digest"
)

func TestChainID(t *testing.T) {
	// To provide a good testing base, we define the individual links in a
	// chain recursively, illustrating the calculations for each chain.
	//
	// Note that we use invalid digests for the unmodified identifiers here to
	// make the computation more readable.
	chainDigestAB := digest.FromString("sha256:a" + " " + "sha256:b")              // chain for A|B
	chainDigestABC := digest.FromString(chainDigestAB.String() + " " + "sha256:c") // chain for A|B|C

	for _, testcase := range []struct {
		Name     string
		Digests  []digest.Digest
		Expected []digest.Digest
	}{
		{
			Name: "nil",
		},
		{
			Name:     "empty",
			Digests:  []digest.Digest{},
			Expected: []digest.Digest{},
		},
		{
			Name:     "identity",
			Digests:  []digest.Digest{"sha256:a"},
			Expected: []digest.Digest{"sha256:a"},
		},
		{
			Name:     "two",
			Digests:  []digest.Digest{"sha256:a", "sha256:b"},
			Expected: []digest.Digest{"sha256:a", chainDigestAB},
		},
		{
			Name:     "three",
			Digests:  []digest.Digest{"sha256:a", "sha256:b", "sha256:c"},
			Expected: []digest.Digest{"sha256:a", chainDigestAB, chainDigestABC},
		},
	} {
		t.Run(testcase.Name, func(t *testing.T) {
			t.Log("before", testcase.Digests)

			var ids []digest.Digest

			if testcase.Digests != nil {
				ids = make([]digest.Digest, len(testcase.Digests))
				copy(ids, testcase.Digests)
			}

			ids = ChainIDs(ids)
			t.Log("after", ids)
			if !reflect.DeepEqual(ids, testcase.Expected) {
				t.Errorf("unexpected chain: %v != %v", ids, testcase.Expected)
			}

			if len(testcase.Digests) == 0 {
				return
			}

			// Make sure parent stays stable
			if ids[0] != testcase.Digests[0] {
				t.Errorf("parent changed: %v != %v", ids[0], testcase.Digests[0])
			}

			// make sure that the ChainID function takes the last element
			id := ChainID(testcase.Digests)
			if id != ids[len(ids)-1] {
				t.Errorf("incorrect chain id returned from ChainID: %v != %v", id, ids[len(ids)-1])
			}
		})
	}
}
