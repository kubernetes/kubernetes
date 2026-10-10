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
	"bytes"
	"errors"
	"runtime/pprof"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
)

// leakProfile encodes samples the way runtime/pprof writes the goroutineleak
// profile. Each sample is a count and a stack of function names, innermost
// first.
func leakProfile(t *testing.T, sampleType string, samples ...sample) []byte {
	t.Helper()
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: sampleType, Unit: "count"}},
		PeriodType: &profile.ValueType{Type: sampleType, Unit: "count"},
		Period:     1,
	}
	functions := map[string]*profile.Function{}
	for _, s := range samples {
		ps := &profile.Sample{Value: []int64{s.count}}
		for _, name := range s.stack {
			fn, ok := functions[name]
			if !ok {
				fn = &profile.Function{ID: uint64(len(functions) + 1), Name: name, Filename: "/src/" + name + ".go"}
				functions[name] = fn
				p.Function = append(p.Function, fn)
			}
			loc := &profile.Location{ID: uint64(len(p.Location) + 1), Line: []profile.Line{{Function: fn, Line: 13}}}
			p.Location = append(p.Location, loc)
			ps.Location = append(ps.Location, loc)
		}
		p.Sample = append(p.Sample, ps)
	}
	var buf bytes.Buffer
	if err := p.Write(&buf); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	return buf.Bytes()
}

type sample struct {
	count int64
	stack []string
}

func TestParseProfile(t *testing.T) {
	body := leakProfile(t, "goroutineleak",
		sample{1, []string{"runtime.gopark", "runtime.chansend", "main.leakOnSend", "runtime.goexit"}},
		sample{3, []string{"runtime.gopark", "runtime.chanrecv", "main.leakForever", "runtime.goexit"}},
		sample{2, []string{"runtime.gopark", "runtime.goexit"}},
	)
	res, err := Parse("kube-apiserver", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Total != 6 {
		t.Errorf("Total = %d, want 6", res.Total)
	}
	// Sorted most frequent first, runtime frames skipped unless nothing else is left.
	want := []Leak{
		{Count: 3, Function: "main.leakForever", Location: "/src/main.leakForever.go:13"},
		{Count: 2, Function: "runtime.gopark", Location: "/src/runtime.gopark.go:13"},
		{Count: 1, Function: "main.leakOnSend", Location: "/src/main.leakOnSend.go:13"},
	}
	if len(res.Leaks) != len(want) {
		t.Fatalf("Leaks = %+v, want %+v", res.Leaks, want)
	}
	for i := range want {
		if res.Leaks[i] != want[i] {
			t.Errorf("Leaks[%d] = %+v, want %+v", i, res.Leaks[i], want[i])
		}
	}
}

func TestParseEmptyProfile(t *testing.T) {
	res, err := Parse("kube-apiserver", leakProfile(t, "goroutineleak"))
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
// response, for example an error page or a different profile.
func TestParseUnrecognized(t *testing.T) {
	for name, body := range map[string][]byte{
		"empty":        nil,
		"html":         []byte("<html><body>404 page not found</body></html>"),
		"wrongProfile": leakProfile(t, "goroutine", sample{195, []string{"main.worker"}}),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse("kube-apiserver", body); err == nil {
				t.Errorf("expected an error for %q", name)
			}
		})
	}
}

// TestParseRuntimeProfile parses what the Go runtime actually writes, so that
// a change of the profile format fails here instead of hiding leaks.
func TestParseRuntimeProfile(t *testing.T) {
	go leakForTest()

	deadline := time.Now().Add(30 * time.Second)
	for {
		var buf bytes.Buffer
		if err := pprof.Lookup("goroutineleak").WriteTo(&buf, 0); err != nil {
			t.Fatalf("writing profile: %v", err)
		}
		res, err := Parse("test", buf.Bytes())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for _, l := range res.Leaks {
			if strings.HasSuffix(l.Function, ".leakForTest") && strings.Contains(l.Location, "invariants_test.go:") {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("leaked goroutine not found in:\n%s", Report([]Result{res}))
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// leakForTest blocks forever on a channel which nothing else can reach.
func leakForTest() {
	<-make(chan struct{})
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
	for _, want := range []string{"3 leaked goroutine(s)", "foo.run", "foo.go:1", "kube-apiserver"} {
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
		{Component: "kube-system/kube-scheduler-node-1", Err: errors.New("connection refused")},
	})
	if !strings.Contains(report, "kube-apiserver (ok)") {
		t.Errorf("report should list checked components:\n%s", report)
	}
	if !strings.Contains(report, "not checked") {
		t.Errorf("report should list skipped components:\n%s", report)
	}
}

// TestControlPlanePodMatching pins the pod name patterns used to find
// kube-controller-manager and kube-scheduler static pods.
func TestControlPlanePodMatching(t *testing.T) {
	for name, want := range map[string]*struct{ kcm, sched bool }{
		"kube-controller-manager-node-1": {kcm: true},
		"kube-scheduler-node-1":          {sched: true},
		"kube-apiserver-node-1":          {},
		"etcd-node-1":                    {},
		"coredns-abc-def":                {},
	} {
		t.Run(name, func(t *testing.T) {
			if got := regKubeControllerManager.MatchString(name); got != want.kcm {
				t.Errorf("regKubeControllerManager.MatchString(%q) = %v, want %v", name, got, want.kcm)
			}
			if got := regKubeScheduler.MatchString(name); got != want.sched {
				t.Errorf("regKubeScheduler.MatchString(%q) = %v, want %v", name, got, want.sched)
			}
		})
	}
}
