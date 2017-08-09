/*
Copyright 2015 The Kubernetes Authors.

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

// Package rand provides utilities related to randomization.
package rand

import (
	"math/rand"
	"sync"
	"time"
)

var rng = struct {
	sync.Mutex
	rand *rand.Rand
}{
	rand: rand.New(rand.NewSource(time.Now().UTC().UnixNano())),
}

// Intn generates an integer in range [0,max).
// By design this should panic if input is invalid, <= 0.
func Intn(max int) int {
	rng.Lock()
	defer rng.Unlock()
	return rng.rand.Intn(max)
}

// IntnRange generates an integer in range [min,max).
// By design this should panic if input is invalid, <= 0.
func IntnRange(min, max int) int {
	rng.Lock()
	defer rng.Unlock()
	return rng.rand.Intn(max-min) + min
}

// IntnRange generates an int64 integer in range [min,max).
// By design this should panic if input is invalid, <= 0.
func Int63nRange(min, max int64) int64 {
	rng.Lock()
	defer rng.Unlock()
	return rng.rand.Int63n(max-min) + min
}

// Seed seeds the rng with the provided seed.
func Seed(seed int64) {
	rng.Lock()
	defer rng.Unlock()

	rng.rand = rand.New(rand.NewSource(seed))
}

// Perm returns, as a slice of n ints, a pseudo-random permutation of the integers [0,n)
// from the default Source.
func Perm(n int) []int {
	rng.Lock()
	defer rng.Unlock()
	return rng.rand.Perm(n)
}

// We omit vowels from the set of available characters to reduce the chances
// of "bad words" being formed.
var alphanums = []rune("bcdfghjklmnpqrstvwxz2456789")

// String generates a random alphanumeric string, without vowels, which is n
// characters long.  This will panic if n is less than zero.
func String(length int) string {
	b := make([]rune, length)
	for i := range b {
		b[i] = alphanums[Intn(len(alphanums))]
	}
	return string(b)
}
