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
	"net/http"
	"testing"
	"time"

	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/workqueue"
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

	reg.Register(FaultRule{Name: "matched", Match: ClientMatch{Verb: "GET", Resource: "pods"}})
	reg.Register(FaultRule{Name: "never-matched", Match: ClientMatch{Verb: "GET", Resource: "daemonsets"}})
	reg.Register(FaultRule{Name: "optional-unmatched", Match: ClientMatch{Verb: "GET", Resource: "leases"}, Optional: true})

	// Exercise only the "matched" rule's site.
	reg.ResolveTransport(context.Background(), ClientFacts{Verb: "GET", Resource: "pods"})

	unmatched := reg.UnmatchedRules()
	if len(unmatched) != 1 || unmatched[0] != "never-matched" {
		t.Fatalf("UnmatchedRules() = %v, want exactly [never-matched]", unmatched)
	}
}

// TestExpectedRulesNotTriggered verifies the delivery guard: a rule declared
// ExpectTriggered must actually fire, not merely match a site.
func TestExpectedRulesNotTriggered(t *testing.T) {
	reg := NewFaultRegistry()

	reg.Register(FaultRule{Name: "fired", Match: ClientMatch{Verb: "GET"}, Condition: TriggerAlways(), ExpectTriggered: true})
	reg.Register(FaultRule{Name: "matched-never-fired", Match: ClientMatch{Verb: "GET"}, Condition: TriggerOnOccurrence(99), ExpectTriggered: true})
	reg.Register(FaultRule{Name: "not-expected", Match: ClientMatch{Verb: "GET"}, Condition: TriggerOnOccurrence(99)})

	reg.ResolveTransport(context.Background(), ClientFacts{Verb: "GET", Resource: "pods"})

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
	minimal := ControllerProfile{
		Root:       ResourceRef{Resource: "leases", Name: "n"},
		WritesRoot: true,
	}
	full := ControllerProfile{
		Root:             ResourceRef{Group: "apps", Resource: "daemonsets", Name: "ds-1"},
		WritesRootStatus: true,
		UsesExpectations: true,
		Child:            &ChildResource{Resource: "pods", CreatedByController: true},
	}

	for _, p := range []struct {
		name    string
		profile ControllerProfile
	}{{"minimal", minimal}, {"full", full}} {
		t.Run(p.name, func(t *testing.T) {
			reg := NewFaultRegistry()
			for _, scenario := range StandardChaosMatrix() {
				if scenario.Faults == nil {
					continue
				}
				for _, rule := range scenario.Faults(p.profile, reg) {
					if err := validateFaultDomain(rule); err != nil {
						t.Errorf("scenario %q produced an invalid rule %q: %v", scenario.Name, rule.Name, err)
					}
					switch rule.Name {
					case "ChildCacheSyncLag", "ChildCacheLagCombined", "StaleCacheWatchDeath", "FlakyChildWrites":
						if p.profile.Child == nil {
							t.Errorf("scenario %q produced child fault %q for a profile with no child resource", scenario.Name, rule.Name)
						}
					case "ShiftExpectationsClock":
						if !p.profile.UsesExpectations {
							t.Errorf("scenario %q produced %q for a profile without expectations", scenario.Name, rule.Name)
						}
					case "RootStatusConflict", "FlakyRootStatusWrites":
						if !p.profile.WritesRootStatus {
							t.Errorf("scenario %q produced %q for a profile that never writes /status", scenario.Name, rule.Name)
						}
					case "RootConflict", "FlakyRootWrites":
						if !p.profile.WritesRoot {
							t.Errorf("scenario %q produced %q for a profile that never writes the root object", scenario.Name, rule.Name)
						}
					}
				}
			}
		})
	}
}

// TestFaultInjectingWorkQueue verifies that Add, AddRateLimited, AddAfter, and Get trigger queue faults.
func TestFaultInjectingWorkQueue(t *testing.T) {
	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "queue-add-delay",
		Match:     QueueMatch{Queue: "test-queue", Op: "add"},
		Condition: TriggerAlways(),
		Action:    InjectDelay{Duration: time.Millisecond},
	})
	reg.Register(FaultRule{
		Name:      "queue-get-delay",
		Match:     QueueMatch{Queue: "test-queue", Op: "get"},
		Condition: TriggerAlways(),
		Action:    InjectDelay{Duration: time.Millisecond},
	})

	baseQueue := workqueue.NewTypedRateLimitingQueue[any](workqueue.DefaultTypedItemBasedRateLimiter[any]())
	defer baseQueue.ShutDown()
	wrappedQueue := NewFaultInjectingWorkQueue(baseQueue, reg, "test-queue")

	wrappedQueue.Add("item1")
	wrappedQueue.AddRateLimited("item2")
	wrappedQueue.AddAfter("item3", time.Millisecond)

	if hits := reg.GetHitCount("queue-add-delay"); hits != 3 {
		t.Fatalf("expected 3 hits for queue-add-delay, got %d", hits)
	}

	item, _ := wrappedQueue.Get()
	if item != "item1" {
		t.Fatalf("expected item1, got %v", item)
	}
	if hits := reg.GetHitCount("queue-get-delay"); hits != 1 {
		t.Fatalf("expected 1 hit for queue-get-delay, got %d", hits)
	}
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

// TestValidateFaultDomain verifies the typed-verdict guard: an action may only
// be attached to a point whose injection site it can actually serve. A
// mis-targeted fault is rejected at registration; blocking faults and nil
// signals are accepted anywhere.
func TestValidateFaultDomain(t *testing.T) {
	tests := []struct {
		name    string
		rule    FaultRule
		wantErr bool
	}{
		{
			name: "transport fault on client match",
			rule: FaultRule{Match: ClientMatch{Verb: "PUT", Resource: "leases"}, Action: NewHTTPStatusError(409, metav1.StatusReasonConflict, "x")},
		},
		{
			name:    "clock fault on client match is rejected",
			rule:    FaultRule{Match: ClientMatch{Verb: "PUT", Resource: "leases"}, Action: ShiftTime{Offset: time.Minute}},
			wantErr: true,
		},
		{
			name: "cache fault on cache match",
			rule: FaultRule{Match: CacheMatch{Cache: "pod-cache"}, Action: StaleRead{}},
		},
		{
			name:    "transport fault on cache match is rejected",
			rule:    FaultRule{Match: CacheMatch{Cache: "pod-cache"}, Action: NewHTTPStatusError(500, metav1.StatusReasonInternalError, "x")},
			wantErr: true,
		},
		{
			name: "clock fault on clock match",
			rule: FaultRule{Match: ClockMatch{Clock: "expectations"}, Action: ShiftTime{Offset: time.Minute}},
		},
		{
			name: "blocking fault is valid anywhere",
			rule: FaultRule{Match: ClientMatch{Verb: "PUT", Resource: "leases"}, Action: InjectDelay{Duration: time.Millisecond}},
		},
		{
			name: "nil action is valid",
			rule: FaultRule{Match: ClientMatch{Verb: "PUT", Resource: "leases"}, Action: nil},
		},
		{
			name:    "nil match is rejected",
			rule:    FaultRule{Action: StaleRead{}},
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

// TestSyntheticErrorResponseContentType verifies the transport directly sets a
// JSON Content-Type and the expected status code on the synthetic response.
func TestSyntheticErrorResponseContentType(t *testing.T) {
	reg := NewFaultRegistry()
	reg.Register(FaultRule{
		Name:      "conflict",
		Match:     ClientMatch{Verb: "PUT", Resource: "leases", Name: "target-node"},
		Condition: TriggerAlways(),
		Action:    NewHTTPStatusError(http.StatusConflict, metav1.StatusReasonConflict, "object was modified"),
	})

	rt := NewFaultInjectingTransport(&recordingTransport{t: t}, reg, nil)
	req, err := http.NewRequest("PUT", "https://example.test/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/target-node", nil)
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
		Match:     ClientMatch{Verb: "PUT", Resource: "leases", Name: "target-node"},
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
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		t.Fatalf("err = %v (%T), want *apierrors.StatusError", err, err)
	}
	if got := se.ErrStatus.Message; got != "object was modified" {
		t.Errorf("decoded Status.Message = %q, want %q (the Status body was not decoded; is Content-Type set on the synthetic response?)", got, "object was modified")
	}
}
