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

package printers_test

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/printers"
	"testing"
	"time"
)

type stringTestList []struct {
	name     string
	time     time.Time
	expected string
}

func TestElapsedTimeString(t *testing.T) {
	tl := stringTestList{
		{"a while from now", time.Now().Add(2.1e9), "<invalid>"},
		{"almost now", time.Now().Add(1.9e9), "0s"},
		{"now", time.Now(), "0s"},
		{"unknown", time.Time{}, "<unknown>"},
		{"30 seconds ago", time.Now().Add(-3e10), "30s"},
		{"5 minutes ago", time.Now().Add(-3e11), "5m"},
		{"an hour ago", time.Now().Add(-6e12), "1h"},
		{"2 days ago", time.Now().UTC().AddDate(0, 0, -2), "2d"},
		{"months ago", time.Now().UTC().AddDate(0, 0, -90), "90d"},
		{"10 years ago", time.Now().UTC().AddDate(-10, 0, 0), "10y"},
	}
	for _, test := range tl {
		actual := printers.ElapsedTimeString(metav1.Time{Time: test.time})
		if actual != test.expected {
			t.Errorf("On %v, expected '%v', but got '%v'", test.name, test.expected, actual)
		}
	}
}
