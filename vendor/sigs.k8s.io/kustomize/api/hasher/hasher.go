// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package hasher

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
)

// SortArrayAndComputeHash sorts a string array and
// returns a hash for it
func SortArrayAndComputeHash(s []string) (string, error) {
	sort.Strings(s)
	data, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return Encode(Hash(string(data)))
}

// Copied from https://github.com/kubernetes/kubernetes
// /blob/master/pkg/kubectl/util/hash/hash.go
func Encode(hex string) (string, error) {
	if len(hex) < 10 {
		return "", fmt.Errorf(
			"input length must be at least 10")
	}
	enc := []rune(hex[:10])
	for i := range enc {
		switch enc[i] {
		case '0':
			enc[i] = 'g'
		case '1':
			enc[i] = 'h'
		case '3':
			enc[i] = 'k'
		case 'a':
			enc[i] = 'm'
		case 'e':
			enc[i] = 't'
		}
	}
	return string(enc), nil
}

// Hash returns the hex form of the sha256 of the argument.
func Hash(data string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
}
