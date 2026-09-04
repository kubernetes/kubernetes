/*
Copyright The Kubernetes Authors.

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

package goroutineleak

import (
	"errors"
	"strings"
	"testing"
)

// realProfile is verbatim output of GET /debug/pprof/goroutineleak?debug=1
// from a Go 1.27 program with four deliberately leaked goroutines.
const realProfile = `goroutineleak profile: total 4
3 @ 0x48c1aa 0x419c2e 0x419772 0x67d745 0x493041
#	0x67d744	main.leakForever+0x24	/tmp/leaktest/main.go:13

1 @ 0x48c1aa 0x418d1c 0x418917 0x67d788 0x493041
#	0x67d787	main.leakOnSend+0x27	/tmp/leaktest/main.go:19
`

const emptyProfile = `goroutineleak profile: total 0
`

func TestParseRealProfile(t *testing.T) {
	res, err := Parse("kube-apiserver", []byte(realProfile))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Total != 4 {
		t.Errorf("Total = %d, want 4", res.Total)
	}
	if len(res.Leaks) != 2 {
		t.Fatalf("len(Leaks) = %d, want 2", len(res.Leaks))
	}
	// Sorted most frequent first.
	if res.Leaks[0].Count != 3 {
		t.Errorf("Leaks[0].Count = %d, want 3", res.Leaks[0].Count)
	}
	if res.Leaks[0].Function != "main.leakForever+0x24" {
		t.Errorf("Leaks[0].Function = %q", res.Leaks[0].Function)
	}
	if res.Leaks[0].Location != "/tmp/leaktest/main.go:13" {
		t.Errorf("Leaks[0].Location = %q", res.Leaks[0].Location)
	}
	if res.Leaks[1].Count != 1 {
		t.Errorf("Leaks[1].Count = %d, want 1", res.Leaks[1].Count)
	}
}

func TestParseEmptyProfile(t *testing.T) {
	res, err := Parse("kube-apiserver", []byte(emptyProfile))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Total != 0 {
		t.Errorf("Total = %d, want 0", res.Total)
	}
	if len(res.Leaks) != 0 {
		t.Errorf("len(Leaks) = %d, want 0", len(res.Leaks))
	}
}

// TestParseUnrecognized guards against silently accepting a different
// response, for example an HTML error page or a profile format change.
func TestParseUnrecognized(t *testing.T) {
	for name, body := range map[string]string{
		"empty": "",
		"html":  "<html><body>404 page not found</body></html>",
		"wrong profile": `goroutine profile: total 2376
195 @ 0x48a88e 0x4195ee
`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse("kube-apiserver", []byte(body)); err == nil {
				t.Errorf("expected an error for %q", name)
			}
		})
	}
}

func TestFailureIgnoresUnscrapedComponents(t *testing.T) {
	results := []Result{
		{Component: "kube-apiserver", Total: 0},
		{Component: "kubelet/node-1", Err: errors.New("404 page not found")},
	}
	if got := Failure(results); got != "" {
		t.Errorf("expected no failure, got:\n%s", got)
	}
}

func TestFailureReportsLeaks(t *testing.T) {
	results := []Result{
		{Component: "kube-apiserver", Total: 3, Leaks: []Leak{{Count: 3, Function: "foo.run", Location: "foo.go:1"}}},
		{Component: "kubelet/node-1", Total: 0},
	}
	got := Failure(results)
	if got == "" {
		t.Fatal("expected a failure message")
	}
	for _, want := range []string{"3 leaked goroutine(s)", "foo.run", "foo.go:1", "kube-apiserver", "Owners"} {
		if !strings.Contains(got, want) {
			t.Errorf("failure message missing %q:\n%s", want, got)
		}
	}
}

// TestReportListsCheckedComponents ensures a check which examined nothing is
// distinguishable from one which passed.
func TestReportListsCheckedComponents(t *testing.T) {
	report := Report([]Result{
		{Component: "kube-apiserver", Total: 0},
		{Component: "kubelet/node-1", Err: errors.New("connection refused")},
	})
	if !strings.Contains(report, "kube-apiserver (ok)") {
		t.Errorf("report should list checked components:\n%s", report)
	}
	if !strings.Contains(report, "not checked") {
		t.Errorf("report should list skipped components:\n%s", report)
	}
}

func TestDefaultOwnersAreSet(t *testing.T) {
	if defaultOwners.SIG == "" {
		t.Error("defaultOwners.SIG is not set")
	}
	if len(defaultOwners.Owners) == 0 {
		t.Error("defaultOwners.Owners is not set")
	}
}
