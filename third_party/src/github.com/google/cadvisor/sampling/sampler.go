// Copyright 2014 Google Inc. All Rights Reserved.
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

package sampling

import (
	crand "crypto/rand"
	"encoding/binary"
	"math/rand"
)

func init() {
	// NOTE(dengnan): Even if we picked a good random seed,
	// the random number from math/rand is still not cryptographically secure!
	var seed int64
	binary.Read(crand.Reader, binary.LittleEndian, &seed)
	rand.Seed(seed)
}

type Sampler interface {
	Update(d interface{})
	Len() int
	Reset()
	Map(f func(interface{}))

	// Filter() should update in place. Removing elements may or may not
	// affect the statistical behavior of the sampler, i.e. the probability
	// that an observation will be sampled after removing some elements is
	// implementation defined.
	Filter(filter func(interface{}) bool)
}
