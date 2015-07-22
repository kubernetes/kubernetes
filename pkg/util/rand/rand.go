/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

var letters = []rune("abcdefghijklmnopqrstuvwxyz0123456789")
var numLetters = len(letters)
var rng = struct {
	sync.Mutex
	rand *rand.Rand
}{
	rand: rand.New(rand.NewSource(time.Now().UTC().UnixNano())),
}

// String generates a random alphanumeric string n characters long.  This will
// panic if n is less than zero.
func String(n int) string {
	if n < 0 {
		panic("out-of-bounds value")
	}
	b := make([]rune, n)
	rng.Lock()
	defer rng.Unlock()
	for i := range b {
		b[i] = letters[rng.rand.Intn(numLetters)]
	}
	return string(b)
}

// Seed seeds the rng with the provided seed.
func Seed(seed int64) {
	rng.Lock()
	defer rng.Unlock()

	rng.rand = rand.New(rand.NewSource(seed))
}
