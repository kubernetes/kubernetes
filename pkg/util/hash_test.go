/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"hash/adler32"
	"testing"
)

type wheel struct {
	radius uint32
}

type unicycle struct {
	primaryWheel   *wheel
	licencePlateID string
	tags           map[string]string
}

func TestDeepObjectPointer(t *testing.T) {
	// Arrange
	wheel1 := wheel{radius: 17}
	wheel2 := wheel{radius: 22}
	wheel3 := wheel{radius: 17}

	myUni1 := unicycle{licencePlateID: "blah", primaryWheel: &wheel1, tags: map[string]string{"color": "blue", "name": "john"}}
	myUni2 := unicycle{licencePlateID: "blah", primaryWheel: &wheel2, tags: map[string]string{"color": "blue", "name": "john"}}
	myUni3 := unicycle{licencePlateID: "blah", primaryWheel: &wheel3, tags: map[string]string{"color": "blue", "name": "john"}}

	// Run it more than once to verify determinism of hasher.
	for i := 0; i < 100; i++ {
		hasher1 := adler32.New()
		hasher2 := adler32.New()
		hasher3 := adler32.New()
		// Act
		DeepHashObject(hasher1, myUni1)
		hash1 := hasher1.Sum32()
		DeepHashObject(hasher2, myUni2)
		hash2 := hasher2.Sum32()
		DeepHashObject(hasher3, myUni3)
		hash3 := hasher3.Sum32()

		// Assert
		if hash1 == hash2 {
			t.Errorf("hash1 (%d) and hash2(%d) must be different because they have different values for wheel size", hash1, hash2)
		}

		if hash1 != hash3 {
			t.Errorf("hash1 (%d) and hash3(%d) must be the same because although they point to different objects, they have the same values for wheel size", hash1, hash3)
		}
	}
}
