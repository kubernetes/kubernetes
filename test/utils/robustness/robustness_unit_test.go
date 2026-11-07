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

package robustness

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
)

// recordingTransport is a base RoundTripper that fails the test if it is ever
// reached. A short-circuiting fault must never fall through to the real transport.
type recordingTransport struct {
	t      *testing.T
	called bool
}

func (rt *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.called = true
	rt.t.Errorf("real transport was reached for %s %s; the injected fault should have short-circuited it", req.Method, req.URL.Path)
	return nil, http.ErrServerClosed
}

// TestUnmatchedRules verifies a loud no-op guard: a registered fault whose
// match never accepted a real injection site must be reported, while matched
// rules and Optional rules must not be.
func TestUnmatchedRules(t *testing.T) {
	reg := NewFaultRegistry()

	reg.Register(FaultRule{Name: "matched", Match: ClientMatch{Verbs: []string{"GET"}, Resource: "pods"}})
	reg.Register(FaultRule{Name: "never-matched", Match: ClientMatch{Verbs: []string{"GET"}, Resource: "daemonsets"}})
	reg.Register(FaultRule{Name: "optional-unmatched", Match: ClientMatch{Verbs: []string{"GET"}, Resource: "leases"}, Optional: true})

	// Exercise only the "matched" rule's site.
	_ = reg.ResolveTransport(context.Background(), ClientFacts{Verb: "GET", Resource: "pods"})

	unmatched := reg.UnmatchedRules()
	if len(unmatched) != 1 || unmatched[0] != "never-matched" {
		t.Fatalf("UnmatchedRules() = %v, want exactly [never-matched]", unmatched)
	}
}

// TestExpectedRulesNotTriggered verifies the delivery guard: a rule declared
// ExpectTriggered must actually fire, not merely match a site.
func TestExpectedRulesNotTriggered(t *testing.T) {
	reg := NewFaultRegistry()

	reg.Register(FaultRule{Name: "fired", Match: ClientMatch{Verbs: []string{"GET"}}, Condition: TriggerAlways(), ExpectTriggered: true})
	reg.Register(FaultRule{Name: "matched-never-fired", Match: ClientMatch{Verbs: []string{"GET"}}, Condition: TriggerOnOccurrence(99), ExpectTriggered: true})
	reg.Register(FaultRule{Name: "not-expected", Match: ClientMatch{Verbs: []string{"GET"}}, Condition: TriggerOnOccurrence(99)})

	_ = reg.ResolveTransport(context.Background(), ClientFacts{Verb: "GET", Resource: "pods"})

	silent := reg.ExpectedRulesNotTriggered()
	if len(silent) != 1 || silent[0] != "matched-never-fired" {
		t.Fatalf("ExpectedRulesNotTriggered() = %v, want exactly [matched-never-fired]", silent)
	}
}

// TestStandardChaosMatrixHonorsProfile verifies that every scenario derives its
// faults from the controller profile: a minimal profile yields no faults for
// sites the controller does not exercise, a full profile yields rules for all of
// them, and every produced rule passes registration-time domain validation.
func TestStandardChaosMatrixHonorsProfile(t *testing.T) {
	child := &ChildResource{Resource: "pods", CreatedByController: true}
	profiles := map[string]ControllerProfile{
		"minimal":           {Root: ResourceRef{Resource: "leases", Name: "n"}, WritesRoot: true},
		"full":              {Root: ResourceRef{Group: "apps", Resource: "daemonsets", Name: "ds-1"}, WritesRootStatus: true, UsesExpectations: true, Child: child, WatchStall: time.Second},
		"full-no-stall":     {Root: ResourceRef{Group: "apps", Resource: "daemonsets", Name: "ds-1"}, WritesRootStatus: true, UsesExpectations: true, Child: child},
		"child-not-created": {Root: ResourceRef{Resource: "nodes", Name: "n"}, WritesRoot: true, Child: &ChildResource{Resource: "pods"}},
		"expectations-only": {Root: ResourceRef{Resource: "jobs", Name: "j"}, WritesRootStatus: true, UsesExpectations: true},
	}
	// Exact rule-name sets per scenario; an empty set means the cell is skipped.
	want := map[string]map[string][]string{
		"minimal": {
			"WriteConflicts": {"RootConflict"}, "FlakyAPIServer": {"FlakyRootWrites"}, "CombinedChaos": {"RootConflict"},
		},
		"full": {
			"WriteConflicts": {"RootStatusConflict"}, "CacheSyncLag": {"ChildCacheSyncLag"}, "FlakyAPIServer": {"FlakyChildWrites"},
			"CombinedChaos": {"RootStatusConflict", "ChildCacheLagCombined"}, "StaleRootWatch": {"StallRootWatch"}, "StaleChildWatch": {"StallChildWatch"},
			"ExpectationsTimeout": {"ChildCreateObserved", "ChildWatchDeath", "ShiftExpectationsClock"},
		},
		"full-no-stall": {
			"WriteConflicts": {"RootStatusConflict"}, "CacheSyncLag": {"ChildCacheSyncLag"}, "FlakyAPIServer": {"FlakyChildWrites"},
			"CombinedChaos":       {"RootStatusConflict", "ChildCacheLagCombined"},
			"ExpectationsTimeout": {"ChildCreateObserved", "StaleCacheWatchDeath", "ShiftExpectationsClock"},
		},
		"child-not-created": {
			"WriteConflicts": {"RootConflict"}, "CacheSyncLag": {"ChildCacheSyncLag"}, "FlakyAPIServer": {"FlakyRootWrites"},
			"CombinedChaos": {"RootConflict", "ChildCacheLagCombined"}, "ExpectationsTimeout": {"StaleCacheWatchDeath"},
		},
		"expectations-only": {
			"WriteConflicts": {"RootStatusConflict"}, "FlakyAPIServer": {"FlakyRootStatusWrites"}, "CombinedChaos": {"RootStatusConflict"},
			"ExpectationsTimeout": {"ShiftExpectationsClock"},
		},
	}

	for name, profile := range profiles {
		t.Run(name, func(t *testing.T) {
			reg := NewFaultRegistry()
			for _, scenario := range StandardChaosMatrix() {
				if scenario.Faults == nil {
					continue
				}
				var got []string
				for _, rule := range scenario.Faults(profile, reg) {
					if err := validateFaultDomain(rule); err != nil {
						t.Errorf("scenario %q produced an invalid rule %q: %v", scenario.Name, rule.Name, err)
					}
					if rule.Name == "ShiftExpectationsClock" {
						if shift, ok := rule.Action.(ShiftTime); !ok || shift.Offset <= controller.ExpectationsTimeout {
							t.Errorf("scenario %q: %q must shift past controller.ExpectationsTimeout (%v), got %v", scenario.Name, rule.Name, controller.ExpectationsTimeout, rule.Action)
						}
					}
					got = append(got, rule.Name)
				}
				if diff := cmp.Diff(want[name][scenario.Name], got, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("scenario %q rules for profile %q (-want +got):\n%s", scenario.Name, name, diff)
				}
			}
		})
	}
}

// TestStaleReadCoversNamespacedLister verifies StaleRead reaches every lister
// read path, including the namespace index behind namespaced listers.
func TestStaleReadCoversNamespacedLister(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	real := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "ns", Labels: map[string]string{"app": "x"}}}
	if err := real.Add(pod); err != nil {
		t.Fatal(err)
	}

	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "stale",
		Match:     CacheMatch{Cache: "pods"},
		Condition: TriggerAlways(),
		Action:    StaleRead{},
	})
	lister := corelisters.NewPodLister(NewFaultInjectingIndexer(ctx, real, reg, "pods"))

	// Sanity: the unwrapped indexer sees the pod through the same lister path.
	if got, _ := corelisters.NewPodLister(real).Pods("ns").List(labels.Everything()); len(got) != 1 {
		t.Fatalf("control: expected 1 pod from the real indexer, got %d", len(got))
	}

	if got, err := lister.Pods("ns").List(labels.Everything()); err != nil || len(got) != 0 {
		t.Errorf("namespaced List under StaleRead = %d pods, %v; want 0, nil", len(got), err)
	}
	if got, err := lister.List(labels.Everything()); err != nil || len(got) != 0 {
		t.Errorf("cluster List under StaleRead = %d pods, %v; want 0, nil", len(got), err)
	}
	if _, err := lister.Pods("ns").Get("p"); !apierrors.IsNotFound(err) {
		t.Errorf("Get under StaleRead error = %v; want NotFound", err)
	}
	if keys := NewFaultInjectingIndexer(ctx, real, reg, "pods").ListKeys(); len(keys) != 0 {
		t.Errorf("ListKeys under StaleRead = %v; want empty", keys)
	}
	if reg.GetHitCount("stale") == 0 {
		t.Fatal("StaleRead rule never fired")
	}
}

// TestTriggerAfterRuleHit verifies the gate opens only once the sensor rule fires.
func TestTriggerAfterRuleHit(t *testing.T) {
	reg := NewFaultRegistry()
	reg.Register(FaultRule{Name: "sensor", Match: ClientMatch{Verbs: []string{"POST"}, Resource: "pods"}, Condition: TriggerAlways()})
	gated := TriggerAfterRuleHit(reg, "sensor")

	if gated(1) {
		t.Fatal("gated condition fired before the sensor rule hit")
	}
	_ = reg.ResolveTransport(context.Background(), ClientFacts{Verb: "POST", Resource: "pods"})
	if !gated(2) || !gated(3) {
		t.Fatal("gated condition did not fire after the sensor rule hit")
	}
}

// TestTriggerProbabilityIsSeeded verifies equal seeds replay identical decisions
// and that TriggerAnyOf guarantees the first match.
func TestTriggerProbabilityIsSeeded(t *testing.T) {
	roll := func(seed int64) []bool {
		cond := TriggerProbability(NewFaultRegistryWithSeed(seed), 0.5)
		out := make([]bool, 64)
		for i := range out {
			out[i] = cond(i + 1)
		}
		return out
	}
	a, b := roll(42), roll(42)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed produced different decisions at match %d", i+1)
		}
	}
	if c := roll(43); equalBools(a, c) {
		t.Fatal("different seeds produced identical decisions; seed is not applied")
	}

	always := TriggerAnyOf(TriggerOnOccurrence(1), TriggerProbability(NewFaultRegistryWithSeed(1), 0))
	if !always(1) {
		t.Fatal("TriggerAnyOf did not fire on the guaranteed first occurrence")
	}
	if always(2) {
		t.Fatal("TriggerAnyOf fired on a later occurrence with probability 0")
	}
}

func equalBools(a, b []bool) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestProfileValidation verifies that invalid profiles (missing resource, version in group) fail validation.
func TestProfileValidation(t *testing.T) {
	tests := []struct {
		name    string
		profile ControllerProfile
		wantErr bool
	}{
		{
			name: "valid profile",
			profile: ControllerProfile{
				Root: ResourceRef{Group: "apps", Resource: "daemonsets"},
			},
			wantErr: false,
		},
		{
			name: "missing root resource",
			profile: ControllerProfile{
				Root: ResourceRef{Group: "apps"},
			},
			wantErr: true,
		},
		{
			name: "group with version in root is rejected",
			profile: ControllerProfile{
				Root: ResourceRef{Group: "apps/v1", Resource: "daemonsets"},
			},
			wantErr: true,
		},
		{
			name: "group with version in child is rejected",
			profile: ControllerProfile{
				Root:  ResourceRef{Group: "apps", Resource: "daemonsets"},
				Child: &ChildResource{Group: "core/v1", Resource: "pods"},
			},
			wantErr: true,
		},
		{
			name: "child with missing resource is rejected",
			profile: ControllerProfile{
				Root:  ResourceRef{Group: "apps", Resource: "daemonsets"},
				Child: &ChildResource{Group: "apps"},
			},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.profile.validate()
			if tc.wantErr != (err != nil) {
				t.Fatalf("validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// TestAllOfInvariant verifies that AllOf evaluates each invariant and fails if any fail.
func TestAllOfInvariant(t *testing.T) {
	var called1, called2 bool
	inv1 := func(ctx context.Context, c clientset.Interface) error {
		called1 = true
		return nil
	}
	inv2 := func(ctx context.Context, c clientset.Interface) error {
		called2 = true
		return errors.New("failed invariant 2")
	}

	composite := AllOf(inv1, inv2)
	err := composite(context.Background(), nil)
	if err == nil || err.Error() != "failed invariant 2" {
		t.Fatalf("expected error from invariant 2, got %v", err)
	}
	if !called1 || !called2 {
		t.Fatalf("expected both invariants to be called, got called1=%v called2=%v", called1, called2)
	}
}

// TestValidateFaultDomain verifies that an action may only be attached to a
// compatible match domain, while blocking faults and nil actions are accepted anywhere.
func TestValidateFaultDomain(t *testing.T) {
	tests := []struct {
		name    string
		rule    FaultRule
		wantErr bool
	}{
		{
			name: "transport fault on client match",
			rule: FaultRule{Match: ClientMatch{Verbs: []string{"PUT"}, Resource: "leases"}, Action: NewHTTPStatusError(409, metav1.StatusReasonConflict, "x")},
		},
		{
			name:    "clock fault on client match is rejected",
			rule:    FaultRule{Match: ClientMatch{Verbs: []string{"PUT"}, Resource: "leases"}, Action: ShiftTime{Offset: time.Minute}},
			wantErr: true,
		},
		{
			name: "cache fault on cache match",
			rule: FaultRule{Match: CacheMatch{Cache: "pods"}, Action: StaleRead{}},
		},
		{
			name:    "transport fault on cache match is rejected",
			rule:    FaultRule{Match: CacheMatch{Cache: "pods"}, Action: NewHTTPStatusError(500, metav1.StatusReasonInternalError, "x")},
			wantErr: true,
		},
		{
			name: "clock fault on clock match",
			rule: FaultRule{Match: ClockMatch{Clock: "expectations"}, Action: ShiftTime{Offset: time.Minute}},
		},
		{
			name: "blocking fault is valid anywhere",
			rule: FaultRule{Match: ClientMatch{Verbs: []string{"PUT"}, Resource: "leases"}, Action: InjectDelay{Duration: time.Millisecond}},
		},
		{
			name: "nil action is valid",
			rule: FaultRule{Match: ClientMatch{Verbs: []string{"PUT"}, Resource: "leases"}, Action: nil},
		},
		{
			name:    "nil match is rejected",
			rule:    FaultRule{Action: StaleRead{}},
			wantErr: true,
		},
		{
			name: "blocking fault on watch match",
			rule: FaultRule{Match: ClientMatch{Verbs: []string{VerbWatch}, Resource: "leases"}, Action: InjectDelay{Duration: time.Millisecond}},
		},
		{
			name:    "transport fault on watch match is rejected",
			rule:    FaultRule{Match: ClientMatch{Verbs: []string{VerbWatch}, Resource: "leases"}, Action: NewHTTPStatusError(500, metav1.StatusReasonInternalError, "x")},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateFaultDomain(tc.rule)
			if tc.wantErr != (err != nil) {
				t.Fatalf("validateFaultDomain() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// watchRoundTripper serves a canned JSON watch stream for any request.
type watchRoundTripper struct{ stream string }

func (w *watchRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	h := make(http.Header)
	h.Set("Content-Type", "application/json")
	return &http.Response{StatusCode: http.StatusOK, Header: h, Body: io.NopCloser(strings.NewReader(w.stream))}, nil
}

// TestWatchEventStall verifies a VerbWatch BlockingFault stalls the stream at the
// matching event, passes bytes through unchanged, and is invisible to GET rules.
func TestWatchEventStall(t *testing.T) {
	event := func(name string) string {
		return `{"type":"ADDED","object":{"kind":"Lease","apiVersion":"coordination.k8s.io/v1","metadata":{"name":"` + name + `","namespace":"kube-node-lease"}}}` + "\n"
	}
	bookmark := `{"type":"BOOKMARK","object":{"kind":"Lease","apiVersion":"coordination.k8s.io/v1","metadata":{"resourceVersion":"9"}}}` + "\n"
	errEvent := `{"type":"ERROR","object":{"kind":"Status","apiVersion":"v1","status":"Failure","code":410}}` + "\n"
	stream := bookmark + event("other") + event("target") + errEvent + event("target")
	const stall = 150 * time.Millisecond

	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "stall-target",
		Match:     ClientMatch{Verbs: []string{VerbWatch}, Group: "coordination.k8s.io", Resource: "leases", Namespace: "kube-node-lease", Name: "target"},
		Condition: TriggerOnOccurrence(1),
		Action:    InjectDelay{Duration: stall},
	})
	reg.Register(FaultRule{Name: "get-sensor", Match: ClientMatch{Verbs: []string{"GET"}, Resource: "leases"}, Condition: TriggerAlways()})
	// Wildcard-name watch sensor: must see object events only, never bookmarks or errors.
	reg.Register(FaultRule{Name: "any-lease-event", Match: ClientMatch{Verbs: []string{VerbWatch}, Resource: "leases"}, Condition: TriggerAlways()})

	rt := NewFaultInjectingTransport(&watchRoundTripper{stream: stream}, reg, nil)
	req, err := http.NewRequest(http.MethodGet, "http://apiserver/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases?watch=true", nil)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != stream {
		t.Errorf("watch stream was altered:\n got: %q\nwant: %q", got, stream)
	}
	if elapsed := time.Since(start); elapsed < stall {
		t.Errorf("stream was read in %v; the stall should have held it for at least %v", elapsed, stall)
	}
	if hits := reg.GetHitCount("stall-target"); hits != 1 {
		t.Errorf("stall rule hit %d times; want exactly 1 (first matching event)", hits)
	}
	if unmatched := reg.UnmatchedRules(); len(unmatched) != 1 || unmatched[0] != "get-sensor" {
		t.Errorf("expected only the GET rule to be unmatched by watch events, got unmatched %v", unmatched)
	}
	if hits := reg.GetHitCount("any-lease-event"); hits != 3 {
		t.Errorf("wildcard watch sensor hit %d times; want 3 (bookmark and error events must not count)", hits)
	}
}

// TestBlockingFaultIsRecordedBeforeBlocking verifies a stall counts as hit, and
// BlockingFaultDeadline reports its release time, while the block is still in
// progress; the suite relies on this to wait out stalls.
func TestBlockingFaultIsRecordedBeforeBlocking(t *testing.T) {
	const stall = 500 * time.Millisecond
	reg := NewFaultRegistry()
	reg.Register(FaultRule{Name: "stall", Match: ClientMatch{Verbs: []string{VerbWatch}, Resource: "pods"}, Condition: TriggerAlways(), Action: InjectDelay{Duration: stall}})

	done := make(chan struct{})
	start := time.Now()
	go func() {
		defer close(done)
		reg.ResolveWatchEvent(context.Background(), ClientFacts{Verb: VerbWatch, Resource: "pods", Name: "p"})
	}()
	deadline := time.Now().Add(stall)
	for reg.GetHitCount("stall") == 0 {
		if time.Now().After(deadline) {
			t.Fatal("stall was not recorded as a hit while blocking")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := reg.BlockingFaultDeadline(); got.IsZero() || got.Before(start.Add(stall-50*time.Millisecond)) {
		t.Errorf("BlockingFaultDeadline() = %v during the block; want about %v", got, start.Add(stall))
	}
	<-done
	if elapsed := time.Since(start); elapsed < stall {
		t.Errorf("block released after %v; want at least %v", elapsed, stall)
	}
}

// TestOverlappingRulesCountOnlyAppliedHits verifies that when two transport
// rules match the same request only the first applies and is counted as a hit,
// while the second still counts as matched for the no-op guard.
func TestOverlappingRulesCountOnlyAppliedHits(t *testing.T) {
	reg := NewFaultRegistry()
	site := ClientMatch{Verbs: []string{"PUT"}, Resource: "leases"}
	reg.Register(FaultRule{Name: "first", Match: site, Condition: TriggerAlways(), Action: NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "x")})
	reg.Register(FaultRule{Name: "second", Match: site, Condition: TriggerAlways(), Action: NewHTTPStatusError(http.StatusInternalServerError, metav1.StatusReasonInternalError, "x")})

	resp := reg.ResolveTransport(context.Background(), ClientFacts{Verb: "PUT", Resource: "leases"})
	if resp == nil || resp.StatusCode != http.StatusConflict {
		t.Fatalf("ResolveTransport = %v; want the first rule's 409", resp)
	}
	if h := reg.GetHitCount("first"); h != 1 {
		t.Errorf("first rule hits = %d; want 1", h)
	}
	if h := reg.GetHitCount("second"); h != 0 {
		t.Errorf("second rule hits = %d; want 0 (its action was never delivered)", h)
	}
	if unmatched := reg.UnmatchedRules(); len(unmatched) != 0 {
		t.Errorf("both rules matched the site; got unmatched %v", unmatched)
	}
}

// TestSyntheticErrorResponseContentType verifies the transport directly sets a
// JSON Content-Type and the expected status code on the synthetic response.
func TestSyntheticErrorResponseContentType(t *testing.T) {
	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "conflict",
		Match:     ClientMatch{Verbs: []string{"PUT"}, Resource: "leases", Name: "target-node"},
		Condition: TriggerAlways(),
		Action:    NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "object was modified"),
	})

	rt := NewFaultInjectingTransport(&recordingTransport{t: t}, reg, nil)
	req, err := http.NewRequest(http.MethodPut, "https://example.test/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/target-node", nil)
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("StatusCode = %d, want %d", resp.StatusCode, http.StatusConflict)
	}
	if got := resp.Header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json (without it client-go cannot decode the Status)", got)
	}
}

// TestInjectedConflictIsClassified is the end-to-end check for a real client-go
// clientset wrapped with the fault transport surfaces the injected 409 as a
// properly classified *apierrors.StatusError.
func TestInjectedConflictIsClassified(t *testing.T) {
	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "conflict",
		Match:     ClientMatch{Verbs: []string{"PUT"}, Resource: "leases", Name: "target-node"},
		Condition: TriggerAlways(),
		Action:    NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "object was modified"),
	})

	cfg := &restclient.Config{Host: "https://example.test"}
	cfg.Wrap(func(base http.RoundTripper) http.RoundTripper {
		return NewFaultInjectingTransport(&recordingTransport{t: t}, reg, nil)
	})
	cs := clientset.NewForConfigOrDie(cfg)

	lease := &coordv1.Lease{
		ObjectMeta: metav1.ObjectMeta{Name: "target-node", Namespace: v1.NamespaceNodeLease},
	}
	_, err := cs.CoordinationV1().Leases(v1.NamespaceNodeLease).Update(context.Background(), lease, metav1.UpdateOptions{})
	if err == nil {
		t.Fatal("expected an error from the injected conflict fault, got nil")
	}
	if !apierrors.IsConflict(err) {
		t.Errorf("apierrors.IsConflict(err) = false, want true; err = %v (%T)", err, err)
	}
	// The decoded Status payload only survives when Content-Type is set.
	var se *apierrors.StatusError
	if !errors.As(err, &se) {
		t.Fatalf("err = %v (%T), want *apierrors.StatusError", err, err)
	}
	if got := se.ErrStatus.Message; got != "object was modified" {
		t.Errorf("decoded Status.Message = %q, want %q (the Status body was not decoded; is Content-Type set on the synthetic response?)", got, "object was modified")
	}
}
