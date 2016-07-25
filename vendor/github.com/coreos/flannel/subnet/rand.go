// Copyright 2015 flannel authors
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

package subnet

import (
	"math/rand"
	"time"
)

var rnd *rand.Rand

func init() {
	seed := time.Now().UnixNano()
	rnd = rand.New(rand.NewSource(seed))
}

func randInt(lo, hi int) int {
	return lo + int(rnd.Int31n(int32(hi-lo)))
}
