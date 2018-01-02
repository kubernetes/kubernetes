// Copyright 2016 The etcd Authors
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

package stringutil

import "math/rand"

const (
	chars = "abcdefghijklmnopqrstuvwxyz0123456789"
)

// UniqueStrings retruns a slice of randomly generated unique strings.
func UniqueStrings(maxlen uint, n int) []string {
	exist := make(map[string]bool)
	ss := make([]string, 0)

	for len(ss) < n {
		s := randomString(maxlen)
		if !exist[s] {
			exist[s] = true
			ss = append(ss, s)
		}
	}

	return ss
}

// RandomStrings retruns a slice of randomly generated strings.
func RandomStrings(maxlen uint, n int) []string {
	ss := make([]string, 0)
	for i := 0; i < n; i++ {
		ss = append(ss, randomString(maxlen))
	}
	return ss
}

func randomString(l uint) string {
	s := make([]byte, l)
	for i := 0; i < int(l); i++ {
		s[i] = chars[rand.Intn(len(chars))]
	}
	return string(s)
}
