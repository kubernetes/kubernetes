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

package handlers

import (
	"math/rand"
	"net/url"
	"testing"

	fuzz "github.com/google/gofuzz"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestGenerateLink(t *testing.T) {
	testCases := []struct {
		name          string
		requestInfo   *request.RequestInfo
		obj           runtime.Object
		expect        string
		expectErr     bool
		clusterScoped bool
	}{
		{
			name: "obj has more priority than requestInfo",
			requestInfo: &request.RequestInfo{
				Name:      "should-not-use",
				Namespace: "should-not-use",
				Resource:  "pod",
			},
			obj:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "should-use", Namespace: "should-use"}},
			expect:        "/api/v1/should-use/pod/should-use",
			expectErr:     false,
			clusterScoped: false,
		},
		{
			name: "hit errEmptyName",
			requestInfo: &request.RequestInfo{
				Name:      "should-use",
				Namespace: "should-use",
				Resource:  "pod",
			},
			obj:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "should-not-use"}},
			expect:        "/api/v1/should-use/pod/should-use",
			expectErr:     false,
			clusterScoped: false,
		},
		{
			name: "use namespace of requestInfo if obj namespace is empty",
			requestInfo: &request.RequestInfo{
				Name:      "should-not-use",
				Namespace: "should-use",
				Resource:  "pod",
			},
			obj:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "should-use"}},
			expect:        "/api/v1/should-use/pod/should-use",
			expectErr:     false,
			clusterScoped: false,
		},
		{
			name: "hit error",
			requestInfo: &request.RequestInfo{
				Name:      "",
				Namespace: "",
				Resource:  "pod",
			},
			obj:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{}},
			expect:        "name must be provided",
			expectErr:     true,
			clusterScoped: false,
		},
		{
			name: "cluster scoped",
			requestInfo: &request.RequestInfo{
				Name:      "only-name",
				Namespace: "should-not-use",
				Resource:  "pod",
			},
			obj:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{}},
			expect:        "/api/v1/only-name",
			expectErr:     false,
			clusterScoped: true,
		},
	}

	for _, test := range testCases {
		n := ContextBasedNaming{
			SelfLinker:         meta.NewAccessor(),
			SelfLinkPathPrefix: "/api/v1/",
			ClusterScoped:      test.clusterScoped,
		}
		uri, err := n.GenerateLink(test.requestInfo, test.obj)

		if uri != test.expect && err.Error() != test.expect {
			if test.expectErr {
				t.Fatalf("%s: unexpected non-error: %v", test.name, err)
			} else {
				t.Fatalf("%s: expected: %v, but got: %v", test.name, test.expect, uri)
			}
		}
	}
}

func Test_fastURLPathEncode_fuzz(t *testing.T) {
	specialCases := []string{"/", "//", ".", "*", "/abc%"}
	for _, test := range specialCases {
		got := fastURLPathEncode(test)
		u := url.URL{Path: test}
		expected := u.EscapedPath()
		if got != expected {
			t.Errorf("%q did not match %q", got, expected)
		}
	}
	f := fuzz.New().Funcs(
		func(s *string, c fuzz.Continue) {
			*s = randString(c.Rand)
		},
	)
	for i := 0; i < 2000; i++ {
		var test string
		f.Fuzz(&test)

		got := fastURLPathEncode(test)
		u := url.URL{Path: test}
		expected := u.EscapedPath()
		if got != expected {
			t.Errorf("%q did not match %q", got, expected)
		}
	}
}

// Unicode range fuzzer from github.com/google/gofuzz/fuzz.go

type charRange struct {
	first, last rune
}

var unicodeRanges = []charRange{
	{0x00, 0x255},
	{' ', '~'},           // ASCII characters
	{'\u00a0', '\u02af'}, // Multi-byte encoded characters
	{'\u4e00', '\u9fff'}, // Common CJK (even longer encodings)
}

// randString makes a random string up to 20 characters long. The returned string
// may include a variety of (valid) UTF-8 encodings.
func randString(r *rand.Rand) string {
	n := r.Intn(20)
	runes := make([]rune, n)
	for i := range runes {
		runes[i] = unicodeRanges[r.Intn(len(unicodeRanges))].choose(r)
	}
	return string(runes)
}

// choose returns a random unicode character from the given range, using the
// given randomness source.
func (r *charRange) choose(rand *rand.Rand) rune {
	count := int64(r.last - r.first)
	return r.first + rune(rand.Int63n(count))
}
