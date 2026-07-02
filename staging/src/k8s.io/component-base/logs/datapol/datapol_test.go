/*
Copyright 2020 The Kubernetes Authors.

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

package datapol

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2/ktesting"
)

const (
	marker = "hunter2"
)

type withDatapolTag struct {
	Key string `json:"key" datapolicy:"password"`
}

type withExternalType struct {
	Header http.Header `json:"header"`
}

type noDatapol struct {
	Key string `json:"key"`
}

type datapolInMember struct {
	secrets withDatapolTag
}

type datapolInSlice struct {
	secrets []withDatapolTag
}

type datapolInMap struct {
	secrets map[string]withDatapolTag
}

type datapolBehindPointer struct {
	secrets *withDatapolTag
}

func TestValidate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testcases := []struct {
		name      string
		value     interface{}
		expect    []string
		badFilter bool
	}{{
		name:   "Empty password",
		value:  withDatapolTag{},
		expect: []string{},
	}, {
		name: "Non-empty password",
		value: withDatapolTag{
			Key: marker,
		},
		expect: []string{"password"},
	}, {
		name:   "empty external type",
		value:  withExternalType{Header: http.Header{}},
		expect: []string{},
	}, {
		name: "external type",
		value: withExternalType{Header: http.Header{
			"Authorization": []string{"Bearer hunter2"},
		}},
		expect: []string{"password", "token"},
	}, {
		name:      "no datapol tag",
		value:     noDatapol{Key: marker},
		expect:    []string{},
		badFilter: true,
	}, {
		name: "nested",
		value: datapolInMember{
			secrets: withDatapolTag{
				Key: marker,
			},
		},
		expect: []string{"password"},
	}, {
		name: "nested in pointer",
		value: datapolBehindPointer{
			secrets: &withDatapolTag{Key: marker},
		},
		expect: []string{},
	}, {
		name: "nested in slice",
		value: datapolInSlice{
			secrets: []withDatapolTag{{Key: marker}},
		},
		expect: []string{"password"},
	}, {
		name: "nested in map",
		value: datapolInMap{
			secrets: map[string]withDatapolTag{
				"key": {Key: marker},
			},
		},
		expect: []string{"password"},
	}, {
		name: "nested in map but empty",
		value: datapolInMap{
			secrets: map[string]withDatapolTag{
				"key": {},
			},
		},
		expect: []string{},
	}, {
		name: "struct in interface",
		value: struct{ v interface{} }{v: withDatapolTag{
			Key: marker,
		}},
		expect: []string{"password"},
	}, {
		name: "structptr in interface",
		value: struct{ v interface{} }{v: &withDatapolTag{
			Key: marker,
		}},
		expect: []string{},
	}}
	for _, tc := range testcases {
		res := Verify(logger, tc.value)
		if !assert.ElementsMatch(t, tc.expect, res) {
			t.Errorf("Wrong set of tags for %q. expect %v, got %v", tc.name, tc.expect, res)
		}
		if !tc.badFilter {
			formatted := fmt.Sprintf("%v", tc.value)
			if strings.Contains(formatted, marker) != (len(tc.expect) > 0) {
				t.Errorf("Filter decision doesn't match formatted value for %q: tags: %v, format: %s", tc.name, tc.expect, formatted)
			}
		}
	}
}

// The following types use exported fields because Redact mutates via
// reflection, which can only set exported fields.

type redactString struct {
	Token  string `datapolicy:"token"`
	Public string
}

// redactHeader mirrors the StaticPodURLHeader shape (map[string][]string) that
// leaked credentials in kubernetes/kubernetes#140101.
type redactHeader struct {
	StaticPodURLHeader map[string][]string `datapolicy:"token"`
	PublicMap          map[string][]string
}

type redactBytes struct {
	Secret []byte `datapolicy:"secret-key"`
}

type redactStringSlice struct {
	Secrets []string `datapolicy:"password"`
}

type redactNested struct {
	Inner  redactString
	Public string
}

type redactPointer struct {
	Inner *redactString
}

type redactNoTags struct {
	A string
	B int
	C map[string]string
}

// redactSharedMap has an untagged field declared before a tagged field. When
// both alias the same map value, redaction of the tagged field must not be
// skipped by the walk having already visited the map's address.
type redactSharedMap struct {
	Public map[string][]string
	Secret map[string][]string `datapolicy:"token"`
}

// aliasedSharedMap returns a redactSharedMap whose Public and Secret fields
// point at the same underlying map.
func aliasedSharedMap() *redactSharedMap {
	shared := map[string][]string{"Authorization": {"Bearer hunter2"}}
	return &redactSharedMap{Public: shared, Secret: shared}
}

// aliasedSliceMap returns a map whose values all alias the same backing slice,
// so redaction must replace each value independently rather than skipping
// already-seen slice headers.
func aliasedSliceMap() map[string][]string {
	vals := []string{"Bearer secret"}
	return map[string][]string{"A": vals, "B": vals}
}

func TestRedact(t *testing.T) {
	testcases := []struct {
		name   string
		value  interface{}
		expect interface{}
	}{{
		name:   "string field with datapolicy tag is redacted, untagged field preserved",
		value:  &redactString{Token: marker, Public: "visible"},
		expect: &redactString{Token: redacted, Public: "visible"},
	}, {
		name:   "empty tagged string is still redacted",
		value:  &redactString{Token: "", Public: "visible"},
		expect: &redactString{Token: redacted, Public: "visible"},
	}, {
		name: "map[string][]string tagged field preserves keys, redacts values",
		value: &redactHeader{
			StaticPodURLHeader: map[string][]string{
				"Authorization": {"Bearer hunter2"},
				"X-Custom":      {"a", "b"},
			},
			PublicMap: map[string][]string{
				"Accept": {"application/json"},
			},
		},
		expect: &redactHeader{
			StaticPodURLHeader: map[string][]string{
				"Authorization": {redacted},
				"X-Custom":      {redacted},
			},
			PublicMap: map[string][]string{
				"Accept": {"application/json"},
			},
		},
	}, {
		name:   "byte slice tagged field is redacted",
		value:  &redactBytes{Secret: []byte(marker)},
		expect: &redactBytes{Secret: []byte(redacted)},
	}, {
		name:   "string slice tagged field is redacted to single sentinel",
		value:  &redactStringSlice{Secrets: []string{marker, "another"}},
		expect: &redactStringSlice{Secrets: []string{redacted}},
	}, {
		name:   "nested struct with tagged field is redacted, siblings preserved",
		value:  &redactNested{Inner: redactString{Token: marker, Public: "keep"}, Public: "top"},
		expect: &redactNested{Inner: redactString{Token: redacted, Public: "keep"}, Public: "top"},
	}, {
		name:   "tagged field behind pointer is redacted",
		value:  &redactPointer{Inner: &redactString{Token: marker, Public: "keep"}},
		expect: &redactPointer{Inner: &redactString{Token: redacted, Public: "keep"}},
	}, {
		name:   "nil pointer field is left untouched",
		value:  &redactPointer{Inner: nil},
		expect: &redactPointer{Inner: nil},
	}, {
		name:   "struct with no datapolicy tags is a no-op",
		value:  &redactNoTags{A: "a", B: 1, C: map[string]string{"k": "v"}},
		expect: &redactNoTags{A: "a", B: 1, C: map[string]string{"k": "v"}},
	}, {
		name:   "tagged field inside a by-value interface is redacted",
		value:  &struct{ V interface{} }{V: redactString{Token: marker, Public: "keep"}},
		expect: &struct{ V interface{} }{V: redactString{Token: redacted, Public: "keep"}},
	}, {
		name:   "tagged field inside a pointer-in-interface is redacted",
		value:  &struct{ V interface{} }{V: &redactString{Token: marker, Public: "keep"}},
		expect: &struct{ V interface{} }{V: &redactString{Token: redacted, Public: "keep"}},
	}, {
		// Regression: the untagged Public field is walked before the tagged
		// Secret field. Both alias the same map, so if the walk's visited set is
		// shared with value redaction, the tagged field's seen() check would
		// short-circuit and leave the secret unredacted (fail-open). Because the
		// storage is genuinely shared, redacting Secret also redacts Public,
		// which is the safe (fail-closed) direction.
		name:   "map aliased by an untagged field walked before the tagged field is still redacted",
		value:  aliasedSharedMap(),
		expect: &redactSharedMap{Public: map[string][]string{"Authorization": {redacted}}, Secret: map[string][]string{"Authorization": {redacted}}},
	}, {
		// Regression: two map values in a tagged map alias the same []string.
		// Terminal slice redaction replaces the slice header rather than
		// mutating the shared backing array, so it must not be short-circuited
		// by the seen() cycle guard — otherwise the second aliased value keeps
		// its original secret (fail-open leak).
		name:   "tagged map whose values alias the same string slice redacts every value",
		value:  &redactHeader{StaticPodURLHeader: aliasedSliceMap()},
		expect: &redactHeader{StaticPodURLHeader: map[string][]string{"A": {redacted}, "B": {redacted}}},
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			Redact(tc.value)
			assert.Equal(t, tc.expect, tc.value)
		})
	}
}

// TestRedactNilAndNonPointer verifies Redact does not panic on inputs it cannot
// mutate.
func TestRedactNilAndNonPointer(t *testing.T) {
	// nil interface
	Redact(nil)
	// non-pointer struct cannot be mutated but must not panic
	v := redactString{Token: marker}
	Redact(v)
}

// redactCyclic is a self-referential type used to prove Redact does not recurse
// forever on a cyclic object graph. A stack overflow would be a fatal error that
// Redact's recover() cannot catch.
type redactCyclic struct {
	Token string `datapolicy:"token"`
	Self  *redactCyclic
	Peers map[string]*redactCyclic
}

func TestRedactCyclic(t *testing.T) {
	// Pointer cycle: node references itself.
	node := &redactCyclic{Token: marker}
	node.Self = node
	node.Peers = map[string]*redactCyclic{"self": node}

	Redact(node)

	if node.Token != redacted {
		t.Errorf("Token not redacted on cyclic input: got %q", node.Token)
	}
	// The cycle must have been broken (no crash) and the same node reached
	// through the cycle is the already-redacted node.
	if node.Self.Token != redacted {
		t.Errorf("Token not redacted through cycle: got %q", node.Self.Token)
	}
}

// TestRedactCyclicSlice proves a slice that references itself through an
// interface element does not recurse forever. Slices carry identity via their
// backing-array address; without recording that identity the walk would follow
// the slice/interface path until a fatal stack overflow that recover() cannot
// catch.
func TestRedactCyclicSlice(t *testing.T) {
	// Self-referential slice: element 0 holds the slice itself.
	s := make([]interface{}, 1)
	s[0] = s

	// Wrap in a struct so Redact receives an addressable pointer, mirroring how
	// configz hands it a deep-copied runtime object.
	holder := &struct {
		Data []interface{}
	}{Data: s}

	// Must return (break the cycle) rather than crash.
	Redact(holder)
}

// TestRedactAliasedSubslice proves that two slices sharing a backing array but
// spanning different ranges are both fully walked, so a tagged field only
// reachable through the longer slice is still redacted. Keying the visited set
// on the backing-array address alone (without length) would skip the longer
// slice as "already seen" and leak the tagged field.
func TestRedactAliasedSubslice(t *testing.T) {
	type tagged struct {
		Token string `datapolicy:"token"`
	}
	backing := []tagged{{Token: marker}, {Token: marker}}
	holder := &struct {
		Short []tagged
		Full  []tagged
	}{
		Short: backing[:1],
		Full:  backing,
	}

	Redact(holder)

	for i := range backing {
		if backing[i].Token != redacted {
			t.Errorf("Token[%d] not redacted through aliased subslice: got %q", i, backing[i].Token)
		}
	}
}
