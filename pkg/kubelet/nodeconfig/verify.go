/*
Copyright 2017 The Kubernetes Authors.

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

package nodeconfig

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"sort"

	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

const (
	sha256Alg = "sha256"
)

// verifiable is for verifying the integrity of a checkpointed configuration (mainly intended to guard against API object mutation)
type verifiable interface {
	verify() (parsable, error)
}

// verifiableConfigMap is for verifying integrity of config from ConfigMaps
type verifiableConfigMap struct {
	cm *apiv1.ConfigMap
}

// verify checks that re-hashing the data in the ConfigMap results in the hash in the ConfigMap's name.
// If verification fails, returns an error.
// If verification succeeds, returns a parsable. This interface can be used to parse the ConfigMap into a KubeletConfiguration.
func (v *verifiableConfigMap) verify() (parsable, error) {
	name, alg, expectedHash, err := parseConfigName(v.cm.ObjectMeta.Name)
	if err != nil {
		return nil, fmt.Errorf("failed: parse config name %q: %v", err, name)
	}

	actualHash, err := MapStringStringHash(alg, v.cm.Data)
	if err != nil {
		return nil, fmt.Errorf("failed: hash config data: %v", err)
	}

	if actualHash != expectedHash {
		return nil, fmt.Errorf("failed: verify config data: alg: %q, expected hash: %q, actual hash: %q", alg, expectedHash, actualHash)
	}

	return &parsableConfigMap{cm: v.cm}, nil
}

// capture groups:
// nameDash: config name substring, with trailing `-`
// name: config name substring, sans trailing `-`
// alg: algorithm used to produce the hash value
// hash: hash value
const algHashRE = `^(?P<nameDash>(?P<name>[a-z0-9.\-]*){0,1}-){0,1}(?P<alg>[a-z0-9]+)-(?P<hash>[a-f0-9]+)$`

// parseConfigName extracts `name`, `alg`, and `hash` from the name of a configuration
func parseConfigName(n string) (name string, alg string, hash string, err error) {
	alg = ""
	hash = ""

	re, err := regexp.Compile(algHashRE)
	if err != nil {
		return
	}

	// run the regexp, zero matches is treated as an error because it means the name is malformed
	groupNames := re.SubexpNames()
	matches := re.FindStringSubmatch(n)
	if len(matches) == 0 {
		err = fmt.Errorf("name %q did not match regexp %q", n, algHashRE)
		return
	}
	// zip names and matches into a map
	namedMatches := map[string]string{}
	for i, match := range matches {
		namedMatches[groupNames[i]] = match
	}

	name = namedMatches["name"]
	alg = namedMatches["alg"]
	hash = namedMatches["hash"]
	return
}

// MapStringStringHash serializes `m` into a string of pairs, in byte-alphabetic order by key, and takes the hash using `alg`.
// Keys and values are separated with `:` and pairs are separated with `,`. If m is non-empty,
// there is a trailing comma in the pre-hash serialization. If m is empty, there is no trailing comma.
// MapStringStringHash is public because it is used as a utility in some of our tests.
func MapStringStringHash(alg string, m map[string]string) (string, error) {
	// extract key-value pairs from data
	kv := kvPairs(m)
	// sort based on keys
	sort.Slice(kv, func(i, j int) bool {
		return kv[i][0] < kv[j][0]
	})
	// serialize to a string
	s := ""
	for _, p := range kv {
		s = s + p[0] + ":" + p[1] + ","
	}
	return hash(alg, s)
}

// kvPairs extracts the key-value pairs from `m`
func kvPairs(m map[string]string) [][]string {
	kv := make([][]string, len(m))
	i := 0
	for k, v := range m {
		kv[i] = []string{k, v}
		i++
	}
	return kv
}

// hash hashes `data` with `alg` if `alg` is supported
func hash(alg string, data string) (string, error) {
	// take the hash based on alg
	switch alg {
	case sha256Alg:
		sum := sha256.Sum256([]byte(data))
		return fmt.Sprintf("%x", sum), nil
	default:
		return "", fmt.Errorf("requested hash algorithm %q is not supported", alg)
	}
}
