/*
Copyright 2026 The Kubernetes Authors.

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

package dynamicresources

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
)

// A pool with no matching extraction result leaves a zero-value poolExtraction;
// it must park the record as pending rather than settle it with empty state.
func TestCommitPoolLocked_PendingOutcomes(t *testing.T) {
	pool := poolName{driver: "driver.example.com", pool: "p"}
	key := ownerPoolKey{uid: "uid-1", inFlight: true, pool: pool}
	tests := map[string]struct {
		token       uint64
		result      poolExtraction
		want        poolOutcome
		wantPending bool
		wantReason  string
	}{
		"missing result is incomplete": {token: 1, result: poolExtraction{}, want: outcomeWaitEvent, wantPending: true, wantReason: "incomplete"},
		"list error retries":           {token: 1, result: poolExtraction{pool: pool, reason: "listError"}, want: outcomeRetry, wantPending: true, wantReason: "listError"},
		"explicit pending":             {token: 1, result: poolExtraction{pool: pool, pending: true, view: poolMeta{complete: true}}, want: outcomeWaitEvent, wantPending: true, wantReason: "incomplete"},
		"stale token superseded":       {token: 2, result: poolExtraction{pool: pool, view: poolMeta{complete: true}}, want: outcomeSuperseded},
		"complete settles":             {token: 1, result: poolExtraction{pool: pool, view: poolMeta{complete: true}}, want: outcomeSettled},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			a := newAllocatedDevices(logger, &affinityAccounting{})
			claim := &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns", UID: key.uid}}
			rec := &ownerPoolRecord{token: 1, claim: claim}
			a.owners[key] = rec
			rev := a.revision
			a.mutex.Lock()
			got := a.commitPoolLocked(logger, key, tc.token, tc.result)
			a.mutex.Unlock()
			if got != tc.want {
				t.Fatalf("outcome: got %v, want %v", got, tc.want)
			}
			if rec.pending != tc.wantPending || rec.reason != tc.wantReason {
				t.Errorf("record: got pending=%v reason=%q, want pending=%v reason=%q", rec.pending, rec.reason, tc.wantPending, tc.wantReason)
			}
			if tc.want == outcomeSuperseded && a.revision != rev {
				t.Errorf("superseded commit bumped revision %d -> %d", rev, a.revision)
			}
		})
	}
}
