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

package rand

import (
	"math/rand"
	"strings"
	"testing"
)

const (
	maxRangeTestCount = 500
)

func TestString(t *testing.T) {
	valid := "0123456789abcdefghijklmnopqrstuvwxyz"
	for _, l := range []int{0, 1, 2, 10, 123} {
		s := String(l)
		if len(s) != l {
			t.Errorf("expected string of size %d, got %q", l, s)
		}
		for _, c := range s {
			if !strings.ContainsRune(valid, c) {
				t.Errorf("expected valid charaters, got %v", c)
			}
		}
	}
}

var validvowels = "aeiou"
var validconsonants = "bcdfghjkmnpqrstvwxyz"
var validletters = "abcdefghijkmnpqrstuvwxyz"
var validnumbers = "0123456789"

func checkAlternating(t *testing.T, s string) {
	length := len(s)
	if length == 0 {
		return
	}

	phoneticSets := [...]string{validconsonants, validvowels}
	seed := -1
	for i, _ := range phoneticSets {
		if strings.IndexByte(phoneticSets[i], s[0]) != -1 {
			seed = i
			break
		}
	}

	if seed == -1 {
		t.Errorf("expected a valid letter, got %c", s[0])
		return
	}

	for i := 1; i < length; i++ {
		if strings.IndexByte(phoneticSets[(seed+i)%len(phoneticSets)], s[i]) == -1 {
			t.Errorf("expected alternating sequence at position %v, got %v", i, s)
			return
		}
	}
}

func TestPhoneticString(t *testing.T) {
	validnumbers := "0123456789"

	for _, l := range []int{0, 1, 2, 3, 4, 5, 6, 10, 123} {
		s := PhoneticString(l)
		if len(s) != l {
			t.Errorf("expected phonetic string of size %d, got %q", l, s)
		}

		// Split on numbers
		splits := strings.FieldsFunc(s, func(r rune) bool {
			return strings.ContainsRune(validnumbers, r)
		})

		for _, split := range splits {
			if len(split) > 5 {
				t.Errorf("expected string length <= 5 letters, got %q", split)
			}
			checkAlternating(t, split)
		}
	}
}

// Confirm that panic occurs on invalid input.
func TestRangePanic(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Errorf("Panic didn't occur!")
		}
	}()
	// Should result in an error...
	Intn(0)
}

func TestIntn(t *testing.T) {
	// 0 is invalid.
	for _, max := range []int{1, 2, 10, 123} {
		inrange := Intn(max)
		if inrange < 0 || inrange > max {
			t.Errorf("%v out of range (0,%v)", inrange, max)
		}
	}
}

func TestPerm(t *testing.T) {
	Seed(5)
	rand.Seed(5)
	for i := 1; i < 20; i++ {
		actual := Perm(i)
		expected := rand.Perm(i)
		for j := 0; j < i; j++ {
			if actual[j] != expected[j] {
				t.Errorf("Perm call result is unexpected")
			}
		}
	}
}

func TestIntnRange(t *testing.T) {
	// 0 is invalid.
	for min, max := range map[int]int{1: 2, 10: 123, 100: 500} {
		for i := 0; i < maxRangeTestCount; i++ {
			inrange := IntnRange(min, max)
			if inrange < min || inrange >= max {
				t.Errorf("%v out of range (%v,%v)", inrange, min, max)
			}
		}
	}
}

func TestInt63nRange(t *testing.T) {
	// 0 is invalid.
	for min, max := range map[int64]int64{1: 2, 10: 123, 100: 500} {
		for i := 0; i < maxRangeTestCount; i++ {
			inrange := Int63nRange(min, max)
			if inrange < min || inrange >= max {
				t.Errorf("%v out of range (%v,%v)", inrange, min, max)
			}
		}
	}
}
