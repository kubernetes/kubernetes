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
var alphanums = []rune("bcdfghjklmnpqrstvwxz0123456789")
var vowels = []rune("aeiou")
var consonants = []rune("bcdfghjkmnpqrstvwxyz")
var alphabets = []rune("abcdefghijkmnpqrstuvwxyz")
var numbers = []rune("0123456789")

// Returns a random rune from the given rune list
func randElem(set []rune) rune {
	return set[Intn(len(set))]
}

// String generates a random alphanumeric string, without vowels, which is n
// characters long.  This will panic if n is less than zero.
func String(length int) string {
	b := make([]rune, length)
	for i := range b {
		b[i] = randElem(alphanums)
	}
	return string(b)
}

// Returns a rune list of length 'length' with alternating
// vowels and consonants. Randomly starting from any of the two.
// If length is 1, returns any random alphabet
func getAltRune(length int) []rune {
	b := make([]rune, length)
	if length == 1 {
		b[0] = randElem(alphabets)
		return b
	}

	used := 0
	parity := false
	if Intn(2) == 1 && length > 0 {
		// Start with a vowel
		b[used] = randElem(vowels)
		used++
		parity = true
	}

	for ; used < length; used++ {
		if parity {
			b[used] = randElem(consonants)
		} else {
			b[used] = randElem(vowels)
		}
		parity = !parity
	}

	return b
}

// PhoneticString generates a random string compising of alternating
// consonants and vowels, with a number 'near' the middle,
// 'length' characters long. Panics if 'length' is less than zero.
func PhoneticString(length int) string {
	if length < 4 {
		return string(getAltRune(length))
	}

	var b []rune
	numpos := length / 2

	// Whether number should be in center or not
	if Intn(2) == 0 {
		// eg: k7dab or kad7b
		if Intn(2) == 0 {
			numpos--
		} else {
			numpos++
		}
	}

	b = append(getAltRune(numpos), randElem(numbers))
	b = append(b, getAltRune(length-numpos-1)...)

	return string(b)
}
