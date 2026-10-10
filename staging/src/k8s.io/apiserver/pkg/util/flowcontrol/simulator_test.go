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

package flowcontrol

import (
	"container/heap"
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	flowcontrol "k8s.io/api/flowcontrol/v1"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/eventclock"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	fcrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	baseclocktest "k8s.io/utils/clock/testing"
)

const (
	// defaultServerConcurrencyLimit matches the default kube-apiserver flags:
	// --max-requests-inflight=400 + --max-mutating-requests-inflight=200.
	defaultServerConcurrencyLimit = 600

	// bytesPerSeat matches k8s.io/apiserver/pkg/util/flowcontrol/request.bytesPerSeat
	// (KEP-4988 / issue #132233): 1 seat represents 100 KB of memory loaded at once.
	bytesPerSeat = 100_000
)

func TestSimulateAllPriorityLevelsUnderoccupied(t *testing.T) {
	report := simulateAPF(t, 10*time.Second, storage.Stats{}, baseTraffic())

	t.Log("when all priority levels are under-occupied, no requests are rejected or queued")
	for _, pl := range []string{"leader-election", "node-high", "workload-high", "workload-low"} {
		assert.Equal(t, 100.0, report.successPercent(pl), "PL %s should have 100%% success ratio", pl)
		assert.Equal(t, time.Duration(0), report.byPL[pl].p99QueueWait, "PL %s should have 0s max queue wait", pl)
	}
	assert.InDelta(t, 1.4, report.totalConcurencyUtilizationPercent(), 0.001)
}

func TestSimulateOnePriorityLevelOveroccupied_ThrottlesWithIdleServerCapacity(t *testing.T) {
	t.Log("Adding workload-low traffic over the capacity")
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel: "workload-low",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:runaway-tenant",
				Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
			},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     700,
			requestDuration: 100 * time.Millisecond,
		},
	)

	res := simulateAPF(t, 30*time.Second, storage.Stats{}, traffic)

	t.Log("other priority levels in base traffic have 100% success ratio and no queue wait")
	for _, pl := range []string{"leader-election", "node-high", "workload-high"} {
		assert.Equal(t, 100.0, res.successPercent(pl))
		assert.Equal(t, time.Duration(0), res.byPL[pl].p99QueueWait)
	}

	t.Log("even after borrowing converges, ~37% of the 600-seat server remains unlendable and sits idle")
	assert.InDelta(t, 62.9, res.totalConcurencyUtilizationPercent(), 1)

	t.Log("workload-low is throttled to ~52% success ratio and 100ms queue wait despite idle server capacity")
	assert.InDelta(t, 51.7, res.successPercent("workload-low"), 1)
	assert.Equal(t, 100*time.Millisecond, res.byPL["workload-low"].p99QueueWait)
	assert.InDelta(t, 150.2, res.concurrencyLimitUtilizationPercent("workload-low"), 1)
}

func TestSimulateBorrowingConvergenceDelay(t *testing.T) {
	t.Log("Adding workload-low burst (400 callers) that fits once borrowing converges")
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel: "workload-low",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:runaway-tenant",
				Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
			},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     400,
			requestDuration: 100 * time.Millisecond,
		},
	)

	first10s := simulateAPF(t, 10*time.Second, storage.Stats{}, traffic)
	t.Log("for the first 10s (borrowingAdjustmentPeriod), workload-low gets only 23% of its own nominal seats and drops 86% of requests while 90% of the server is idle")
	assert.InDelta(t, 23, first10s.concurrencyLimitUtilizationPercent("workload-low"), 1)
	assert.InDelta(t, 10.1, first10s.totalConcurencyUtilizationPercent(), 1)
	assert.InDelta(t, 14, first10s.successPercent("workload-low"), 1)
	assert.Equal(t, 610*time.Millisecond, first10s.byPL["workload-low"].p99QueueWait)

	converged := simulateAPF(t, 30*time.Second, storage.Stats{}, traffic)
	t.Log("after 30s, borrowing converges and workload-low reaches 150.6% of its nominal seats with 100% success ratio")
	assert.InDelta(t, 150.6, converged.concurrencyLimitUtilizationPercent("workload-low"), 1)
	assert.InDelta(t, 62.9, converged.totalConcurencyUtilizationPercent(), 1)
	assert.Equal(t, 100.0, converged.successPercent("workload-low"))
	assert.Equal(t, 100*time.Millisecond, converged.byPL["workload-low"].p99QueueWait)
}

func TestSimulateFairnessBetweenServiceAccountsWithinPriorityLevel(t *testing.T) {
	sa := func(name string) user.Info {
		return &user.DefaultInfo{
			Name:   "system:serviceaccount:default:" + name,
			Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
		}
	}
	podCreate := request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: "default"}

	t.Log("when first service account needs 40% of seats and second needs 100%, first gets ~40% of seats and second gets ~60%")
	res40 := simulateAPF(t, 30*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-low", user: sa("tenant-1"), requestInfo: podCreate, concurrency: 148, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-low", user: sa("tenant-2"), requestInfo: podCreate, concurrency: 369, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 40, res40.flowSeatSharePercent("workload-low", "system:serviceaccount:default:tenant-1"), 2)
	assert.InDelta(t, 60, res40.flowSeatSharePercent("workload-low", "system:serviceaccount:default:tenant-2"), 2)

	t.Log("when first service account needs 60% of seats and second needs 100%, both get ~50% of seats")
	res60 := simulateAPF(t, 30*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-low", user: sa("tenant-1"), requestInfo: podCreate, concurrency: 221, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-low", user: sa("tenant-2"), requestInfo: podCreate, concurrency: 369, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-low", "system:serviceaccount:default:tenant-1"), 5)
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-low", "system:serviceaccount:default:tenant-2"), 5)
}

func TestSimulateFairnessBetweenNamespacesWithinPriorityLevel(t *testing.T) {
	podCreate := func(ns string) request.RequestInfo {
		return request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: ns}
	}

	t.Log("when first namespace needs 40% of seats and second needs 100%, first gets ~40% of seats and second gets ~60%")
	res40 := simulateAPF(t, 10*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-a"), concurrency: 49, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-b"), concurrency: 122, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 40, res40.flowSeatSharePercent("workload-high", "ns-a"), 2)
	assert.InDelta(t, 60, res40.flowSeatSharePercent("workload-high", "ns-b"), 2)

	t.Log("when first namespace needs 60% of seats and second needs 100%, both get ~50% of seats")
	res60 := simulateAPF(t, 10*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-a"), concurrency: 73, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-b"), concurrency: 122, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-a"), 1)
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-b"), 1)
}

func TestSimulateFairnessBetweenFastAndSlowRequestsWithinPriorityLevel(t *testing.T) {
	podCreate := func(ns string) request.RequestInfo {
		return request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: ns}
	}

	t.Log("when fast flow (10ms) needs 40% of seats and slow flow (100ms) needs 100%, fast gets ~40% of seats and slow gets ~60%")
	res40 := simulateAPF(t, 10*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-a"), concurrency: 49, requestDuration: 10 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-b"), concurrency: 122, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 40, res40.flowSeatSharePercent("workload-high", "ns-a"), 3)
	assert.InDelta(t, 60, res40.flowSeatSharePercent("workload-high", "ns-b"), 3)

	t.Log("when fast flow (10ms) needs 60% of seats and slow flow (100ms) needs 100%, both get ~50% of seats and fast completes ~10x as many requests")
	res60 := simulateAPF(t, 10*time.Second, storage.Stats{}, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-a"), concurrency: 73, requestDuration: 10 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate("ns-b"), concurrency: 122, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-a"), 2)
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-b"), 2)
	assert.InDelta(t, 10, float64(res60.byFlow["ns-a"].completedRequests)/float64(res60.byFlow["ns-b"].completedRequests), 1)
}

func TestSimulateFairnessBetweenDifferentRequestWidthsWithinPriorityLevel(t *testing.T) {
	podStats := storage.Stats{
		ObjectCount:                     100,
		EstimatedAverageObjectSizeBytes: 10_000,
	}
	podCreate := request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: "ns-a"}
	podList := request.RequestInfo{IsResourceRequest: true, Verb: "list", Resource: "pods", Namespace: "ns-b"}

	t.Log("when 1-seat create flow needs 40% of seats and 10-seat 1MB list flow needs 100%, create flow gets ~40% of seats and list flow gets ~60%")
	res40 := simulateAPF(t, 10*time.Second, podStats, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate, concurrency: 45, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podList, concurrency: 12, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 40, res40.flowSeatSharePercent("workload-high", "ns-a"), 5)
	assert.InDelta(t, 60, res40.flowSeatSharePercent("workload-high", "ns-b"), 5)

	t.Log("when 1-seat create flow needs 60% of seats and 10-seat 1MB list flow needs 100%, both get ~50% of seats")
	res60 := simulateAPF(t, 10*time.Second, podStats, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate, concurrency: 68, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podList, concurrency: 12, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-a"), 5)
	assert.InDelta(t, 50, res60.flowSeatSharePercent("workload-high", "ns-b"), 5)
}

func TestSimulateExpensiveRequestsDoNotStarveCheapRequestsWithinPriorityLevel(t *testing.T) {
	// Each 10 MB LIST costs 100 real seats, matching workload-high's ~98-seat nominal limit.
	podStats := storage.Stats{
		ObjectCount:                     1000,
		EstimatedAverageObjectSizeBytes: 10_000,
	}
	podCreate := request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: "ns-a"}
	podList := request.RequestInfo{IsResourceRequest: true, Verb: "list", Resource: "pods", Namespace: "ns-b"}

	t.Log("when 1-seat create flow needs 40% of seats (45 seats) and 100-seat 10MB list flow floods the priority level, create flow gets ~36 seats (~3,625 reqs) while list flow gets ~476 real seats (~475 reqs, 91.5% of real seats) due to maxSeats=15 clamping")
	res40 := simulateAPF(t, 10*time.Second, podStats, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate, concurrency: 45, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podList, concurrency: 20, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 36, res40.byFlow["ns-a"].avgSeats, 3)
	assert.InDelta(t, 476, res40.byFlow["ns-b"].avgSeats, 20)
	assert.InDelta(t, 3625, res40.byFlow["ns-a"].completedRequests, 300)
	assert.InDelta(t, 475, res40.byFlow["ns-b"].completedRequests, 20)
	assert.InDelta(t, 7, res40.flowSeatSharePercent("workload-high", "ns-a"), 2)
	assert.InDelta(t, 91.5, res40.flowSeatSharePercent("workload-high", "ns-b"), 2)

	t.Log("when both flows demand 100% of seats, maxSeats=15 splits APF seats 50/50 (~55 APF seats each) so neither starves, but in real seats create flow gets ~55 seats (~5,519 reqs, 13% of real seats) while list flow gets ~364 real seats (~363 reqs, 85% of real seats)")
	res100 := simulateAPF(t, 10*time.Second, podStats, append(baseTraffic(),
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podCreate, concurrency: 113, requestDuration: 100 * time.Millisecond},
		trafficShape{priorityLevel: "workload-high", user: testUser{name: user.KubeControllerManager}, requestInfo: podList, concurrency: 20, requestDuration: 100 * time.Millisecond},
	))
	assert.InDelta(t, 55, res100.byFlow["ns-a"].avgSeats, 3)
	assert.InDelta(t, 364, res100.byFlow["ns-b"].avgSeats, 20)
	assert.InDelta(t, 5519, res100.byFlow["ns-a"].completedRequests, 300)
	assert.InDelta(t, 363, res100.byFlow["ns-b"].completedRequests, 20)
	assert.InDelta(t, 13, res100.flowSeatSharePercent("workload-high", "ns-a"), 2)
	assert.InDelta(t, 85, res100.flowSeatSharePercent("workload-high", "ns-b"), 2)
}

func TestSimulateExemptTrafficIsNeverThrottledAndReducesNonExemptCapacity(t *testing.T) {
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel:   "exempt",
			user:            &user.DefaultInfo{Name: "admin", Groups: []string{user.SystemPrivilegedGroup, user.AllAuthenticated}},
			requestInfo:     request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: "default"},
			concurrency:     250,
			requestDuration: 100 * time.Millisecond,
		},
		trafficShape{
			priorityLevel: "workload-low",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:runaway-tenant",
				Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
			},
			requestInfo:     request.RequestInfo{IsResourceRequest: true, Verb: "create", Resource: "pods", Namespace: "default"},
			concurrency:     300,
			requestDuration: 100 * time.Millisecond,
		},
	)

	res := simulateAPF(t, 30*time.Second, storage.Stats{}, traffic)

	t.Log("exempt traffic is never queued or rejected and occupies all 250 requested seats")
	assert.Equal(t, 100.0, res.successPercent("exempt"))
	assert.Equal(t, time.Duration(0), res.byPL["exempt"].p99QueueWait)
	assert.InDelta(t, 250, res.byPL["exempt"].avgSeats, 1)

	t.Log("exempt concurrency reduces the seats available to non-exempt workload-low from 150.6% down to ~50% of its nominal limit")
	assert.InDelta(t, 50, res.concurrencyLimitUtilizationPercent("workload-low"), 5)
}

func TestSimulateMultiplePriorityLevelsOveroccupied_WithBorrowing(t *testing.T) {
	t.Log("Adding workload-high and global-default traffic over the capacity (borrowing from idle workload-low)")
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel: "workload-high",
			user:          testUser{name: user.KubeControllerManager},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     300,
			requestDuration: 100 * time.Millisecond,
		},
		trafficShape{
			priorityLevel: "global-default",
			user:          testUser{name: "developer"},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     300,
			requestDuration: 100 * time.Millisecond,
		},
	)

	res := simulateAPF(t, 30*time.Second, storage.Stats{}, traffic)

	t.Log("unaffected priority levels in base traffic remain at 100% success and 0s queue wait")
	for _, pl := range []string{"leader-election", "node-high", "workload-low"} {
		assert.Equal(t, 100.0, res.successPercent(pl))
		assert.Equal(t, time.Duration(0), res.byPL[pl].p99QueueWait)
	}
	t.Log("When borrowing from idle priority levels, workload-high (40 shares) and global-default (20 shares) receive nearly identical seats (~1:1)")
	assert.InDelta(t, float64(res.byPL["workload-high"].avgSeats), float64(res.byPL["global-default"].avgSeats), 10)
	assert.InDelta(t, 216, res.concurrencyLimitUtilizationPercent("workload-high"), 5)
	assert.InDelta(t, 422, res.concurrencyLimitUtilizationPercent("global-default"), 10)
	assert.Equal(t, 100*time.Millisecond, res.byPL["workload-high"].p99QueueWait)
	assert.Equal(t, 100*time.Millisecond, res.byPL["global-default"].p99QueueWait)

	t.Log("server executes ~70.5% of 600 seats on average while throttling workload-high and global-default")
	assert.InDelta(t, 70.5, res.totalConcurencyUtilizationPercent(), 1)
}

func TestSimulateMultiplePriorityLevelsOveroccupied_WithoutBorrowing(t *testing.T) {
	t.Log("Adding workload-high and workload-low traffic over the capacity (workload-low reclaims its lent seats)")
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel: "workload-high",
			user:          testUser{name: user.KubeControllerManager},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     300,
			requestDuration: 100 * time.Millisecond,
		},
		trafficShape{
			priorityLevel: "workload-low",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:runaway-tenant",
				Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
			},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:     300,
			requestDuration: 100 * time.Millisecond,
		},
	)

	res := simulateAPF(t, 30*time.Second, storage.Stats{}, traffic)

	t.Log("unaffected priority levels in base traffic remain at 100% success and 0s queue wait")
	for _, pl := range []string{"leader-election", "node-high"} {
		assert.Equal(t, 100.0, res.successPercent(pl))
		assert.Equal(t, time.Duration(0), res.byPL[pl].p99QueueWait)
	}

	t.Log("When workload-low is also over-occupied and cannot lend its 90% lendable seats, workload-low occupies all 245 of its nominal seats (100 shares) vs workload-high's 172 seats (40 shares)")
	assert.InDelta(t, 100, res.concurrencyLimitUtilizationPercent("workload-low"), 1)
	assert.InDelta(t, 175.5, res.concurrencyLimitUtilizationPercent("workload-high"), 1)
	assert.Equal(t, 100*time.Millisecond, res.byPL["workload-low"].p99QueueWait)
	assert.Equal(t, 100*time.Millisecond, res.byPL["workload-high"].p99QueueWait)

	t.Log("server executes ~70% of 600 seats on average while throttling workload-high harder than workload-low")
	assert.InDelta(t, 69.9, res.totalConcurencyUtilizationPercent(), 1)
}

func TestSimulateMultiplePriorityLevelsOveroccupied_HeavyListsOverloadServerAndCollapseShares(t *testing.T) {
	podStats := storage.Stats{
		ObjectCount:                     1000,
		EstimatedAverageObjectSizeBytes: 10_000,
	}

	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel:   "global-default",
			user:            testUser{name: "developer"},
			requestInfo:     request.RequestInfo{IsResourceRequest: true, Verb: "list", Resource: "pods", Namespace: "default"},
			concurrency:     15,
			requestDuration: time.Second,
		},
	)

	res := simulateAPF(t, 30*time.Second, podStats, traffic)

	t.Log("a single low-priority level (global-default, 20 shares = 49 nominal seats) consumes 1,500 real seats (150 MB = 3,061% of its nominal limit) because maxSeats clamps each 100-seat LIST")
	assert.InDelta(t, 3061, res.concurrencyLimitUtilizationPercent("global-default"), 1)

	t.Log("total server concurrency reaches ~251% (1,508 real seats vs the 600-seat server limit)")
	assert.InDelta(t, 251.4, res.totalConcurencyUtilizationPercent(), 1)
}

func TestSimulateMultiplePriorityLevelsOveroccupied_HeavyListsEraseShareRatio(t *testing.T) {
	podStats := storage.Stats{
		ObjectCount:                     1000,
		EstimatedAverageObjectSizeBytes: 10_000,
	}

	t.Log("Adding identical 10 MB LIST traffic (40 callers, 100 real seats per LIST) to workload-high (40 shares) and global-default (20 shares)")
	traffic := append(baseTraffic(),
		trafficShape{
			priorityLevel:   "workload-high",
			user:            testUser{name: user.KubeControllerManager},
			requestInfo:     request.RequestInfo{IsResourceRequest: true, Verb: "list", Resource: "pods", Namespace: "default"},
			concurrency:     40,
			requestDuration: time.Second,
		},
		trafficShape{
			priorityLevel:   "global-default",
			user:            testUser{name: "developer"},
			requestInfo:     request.RequestInfo{IsResourceRequest: true, Verb: "list", Resource: "pods", Namespace: "default"},
			concurrency:     40,
			requestDuration: time.Second,
		},
	)

	res := simulateAPF(t, 30*time.Second, podStats, traffic)

	t.Log("because maxSeats scales with each priority level's nominal limit (8 seats for global-default vs 15 seats for workload-high), global-default is charged half as many seats per 10 MB LIST and consumes identical real concurrency (~1,800 seats each)")
	assert.InDelta(t, float64(res.byPL["workload-high"].avgSeats), float64(res.byPL["global-default"].avgSeats), 5)
	assert.InDelta(t, 1838, res.concurrencyLimitUtilizationPercent("workload-high"), 5)
	assert.InDelta(t, 3673, res.concurrencyLimitUtilizationPercent("global-default"), 5)

	t.Log("total server concurrency reaches ~601% (3,608 real seats vs the 600-seat server limit)")
	assert.InDelta(t, 601.4, res.totalConcurencyUtilizationPercent(), 1)
}

func baseTraffic() []trafficShape {
	return []trafficShape{
		{
			priorityLevel: "leader-election",
			user:          testUser{name: user.KubeControllerManager},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "update",
				APIGroup:          "coordination.k8s.io",
				Resource:          "leases",
				Namespace:         "kube-system",
				Name:              "kube-controller-manager",
			},
			concurrency:         2,
			requestDuration:     10 * time.Millisecond,
			waitBetweenRequests: 40 * time.Millisecond,
		},
		{
			priorityLevel: "node-high",
			user:          &user.DefaultInfo{Name: "system:node:node-1", Groups: []string{user.NodesGroup, user.AllAuthenticated}},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "patch",
				Resource:          "nodes",
				Subresource:       "status",
				Name:              "node-1",
			},
			concurrency:         20,
			requestDuration:     10 * time.Millisecond,
			waitBetweenRequests: 90 * time.Millisecond,
		},
		{
			priorityLevel: "workload-high",
			user:          testUser{name: user.KubeScheduler},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "create",
				Resource:          "pods",
				Subresource:       "binding",
				Namespace:         "default",
			},
			concurrency:         10,
			requestDuration:     10 * time.Millisecond,
			waitBetweenRequests: 40 * time.Millisecond,
		},
		{
			priorityLevel: "workload-low",
			user: &user.DefaultInfo{
				Name:   "system:serviceaccount:default:app",
				Groups: []string{serviceaccount.AllServiceAccountsGroup, user.AllAuthenticated},
			},
			requestInfo: request.RequestInfo{
				IsResourceRequest: true,
				Verb:              "get",
				Resource:          "pods",
				Namespace:         "default",
			},
			concurrency:         20,
			requestDuration:     10 * time.Millisecond,
			waitBetweenRequests: 40 * time.Millisecond,
		},
	}
}

// =============================================================================
// 2. Default APF Simulation Runner
// =============================================================================

type trafficShape struct {
	priorityLevel       string // Expected priority level routed by default FlowSchemas.
	user                user.Info
	requestInfo         request.RequestInfo
	concurrency         int
	requestDuration     time.Duration
	waitBetweenRequests time.Duration
}

type priorityLevelReport struct {
	nominalConcurrencyLimit int
	completedRequests       int
	rejectedRequests        int
	p99QueueWait            time.Duration
	avgSeats                float64

	queueWaits      []time.Duration
	seatNanoseconds int64
}

type simulationReport struct {
	avgRealSeats float64
	byPL         map[string]*priorityLevelReport
	byFlow       map[string]*priorityLevelReport

	realSeatNanos int64
}

func (s *simulationReport) successPercent(pl string) float64 {
	return float64(s.byPL[pl].completedRequests) / (float64(s.byPL[pl].completedRequests) + float64(s.byPL[pl].rejectedRequests)) * 100
}

func (s *simulationReport) flowSeatSharePercent(pl, flow string) float64 {
	return s.byFlow[flow].avgSeats / s.byPL[pl].avgSeats * 100
}

// concurrencyLimitUtilizationPercent returns the ratio of average real seats used by a priority level
// to its configured nominal concurrency limit (nominalCL).
func (s *simulationReport) concurrencyLimitUtilizationPercent(pl string) float64 {
	return s.byPL[pl].avgSeats / float64(s.byPL[pl].nominalConcurrencyLimit) * 100
}

// totalConcurencyUtilizationPercent returns the ratio of average real seats used across all
// priority levels to defaultServerConcurrencyLimit (600 seats).
func (s *simulationReport) totalConcurencyUtilizationPercent() float64 {
	return s.avgRealSeats / float64(defaultServerConcurrencyLimit) * 100
}

// simulateAPF wires the real APF *configController with the standard
// bootstrap PriorityLevelConfigurations and FlowSchemas, drives it with a
// discrete-event clock (zero wall-clock sleeps), and tracks both APF seats and
// "real" concurrent seats (1 seat = 100 KB of memory loaded at once).
func simulateAPF(t *testing.T, duration time.Duration, listStats storage.Stats, traffic []trafficShape) *simulationReport {
	t.Helper()
	metrics.Register()

	startTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)
	ctlr := newBootstrapAPFController(clk, defaultServerConcurrencyLimit)

	// Measure steady-state averages after the first borrowingAdjustmentPeriod (10s)
	// if the simulation runs long enough for dynamic borrowing to rebalance.
	warmup := time.Duration(0)
	if duration >= 20*time.Second {
		warmup = borrowingAdjustmentPeriod
	}
	measureStart := startTime.Add(warmup)
	endTime := startTime.Add(duration)

	res := &simulationReport{
		byPL:   make(map[string]*priorityLevelReport),
		byFlow: make(map[string]*priorityLevelReport),
	}
	for name, state := range ctlr.priorityLevelStates {
		res.byPL[name] = &priorityLevelReport{
			nominalConcurrencyLimit: state.nominalCL,
		}
	}
	flowReport := func(flow string) *priorityLevelReport {
		if r, ok := res.byFlow[flow]; ok {
			return r
		}
		r := &priorityLevelReport{}
		res.byFlow[flow] = r
		return r
	}

	var (
		lastSampleTime     = startTime
		inFlightAPFSeats   int64
		inFlightRealSeats  int64
		inFlightRealByPL   = make(map[string]int64)
		inFlightRealByFlow = make(map[string]int64)
	)

	integrate := func() {
		now := clk.Now()
		dt := now.Sub(lastSampleTime)
		prev := lastSampleTime
		lastSampleTime = now
		if dt <= 0 || !now.After(measureStart) {
			return
		}
		if prev.Before(measureStart) {
			dt = now.Sub(measureStart)
		}
		nanos := int64(dt)
		res.realSeatNanos += inFlightRealSeats * nanos
		for pl, st := range res.byPL {
			st.seatNanoseconds += inFlightRealByPL[pl] * nanos
		}
		for flow, fr := range res.byFlow {
			fr.seatNanoseconds += inFlightRealByFlow[flow] * nanos
		}
	}

	verifyQueueSetFidelity := func() {
		var qsSeats int64
		for _, state := range ctlr.priorityLevelStates {
			qsSeats += int64(state.queues.Dump(false).SeatsInUse)
		}
		require.Equal(t, qsSeats, inFlightAPFSeats, "simulator fidelity mismatch at t=%s", clk.Now().Sub(startTime))
	}

	estimator := fcrequest.NewWorkEstimator(
		func(string) (storage.Stats, error) { return listStats, nil },
		func(*request.RequestInfo) int { return 0 },
		fcrequest.DefaultWorkEstimatorConfig(),
		ctlr.GetMaxSeats,
	)

	// Run APF's periodic borrowing adjustment loop every 10s of simulated time.
	var scheduleBorrowing func()
	scheduleBorrowing = func() {
		clk.EventAfterDuration(func(now time.Time) {
			if !now.Before(endTime) {
				return
			}
			ctlr.updateBorrowing()
			scheduleBorrowing()
		}, borrowingAdjustmentPeriod)
	}
	scheduleBorrowing()

	for _, shape := range traffic {
		s := shape
		reqInfo := s.requestInfo
		digest := RequestDigest{
			RequestInfo: &reqInfo,
			User:        s.user,
		}
		// Use limit=<ObjectCount> so LISTs are evaluated as uncached etcd reads
		// loading all returned objects into memory at once (issue #132233).
		rawQuery := ""
		reqMemoryBytes := int64(bytesPerSeat)
		if reqInfo.Verb == "list" && listStats.ObjectCount > 0 {
			rawQuery = fmt.Sprintf("limit=%d", listStats.ObjectCount)
			reqMemoryBytes = listStats.ObjectCount * listStats.EstimatedAverageObjectSizeBytes
		}
		reqRealSeats := max(int64(1), reqMemoryBytes/bytesPerSeat)
		httpReq := (&http.Request{URL: &url.URL{RawQuery: rawQuery}}).WithContext(
			request.WithRequestInfo(context.Background(), &reqInfo),
		)

		for w := 0; w < s.concurrency; w++ {
			var issueNext func(time.Time)
			issueNext = func(now time.Time) {
				if !now.Before(endTime) {
					return
				}
				arrival := clk.Now()
				var matchedFS, matchedPL, matchedFlow string
				var est fcrequest.WorkEstimate
				var queuedAt time.Time
				var inQueueNow bool
				var fqReq fq.Request
				var startExecution func()

				noteFn := func(fs *flowcontrol.FlowSchema, plc *flowcontrol.PriorityLevelConfiguration, flowDistinguisher string) {
					matchedFS = fs.Name
					matchedPL = plc.Name
					matchedFlow = flowDistinguisher
				}
				workEstr := func() fcrequest.WorkEstimate {
					// Use the real APF WorkEstimator and map requestDuration to
					// AdditionalLatency so QueueSet holds the seats on the single-threaded event loop.
					est = estimator.EstimateWork(httpReq, matchedFS, matchedPL)
					est.FinalSeats = est.InitialSeats
					est.AdditionalLatency = s.requestDuration
					return est
				}
				queueNoteFn := func(inQueue bool) {
					if inQueue {
						inQueueNow = true
						queuedAt = clk.Now()
					} else {
						inQueueNow = false
						if !queuedAt.IsZero() && !clk.Now().Before(measureStart) {
							if wait := clk.Now().Sub(queuedAt); wait > res.byPL[matchedPL].p99QueueWait {
								res.byPL[matchedPL].p99QueueWait = wait
							}
						}
						if fqReq != nil {
							clk.EventAfterDuration(func(time.Time) { startExecution() }, 0)
						}
					}
				}

				_, _, _, fqReq, _ = ctlr.startRequest(context.Background(), digest, noteFn, workEstr, queueNoteFn)
				if s.priorityLevel != "" {
					require.Equal(t, s.priorityLevel, matchedPL, "traffic shape routed to unexpected priority level")
				}
				if fqReq == nil {
					if !clk.Now().Before(measureStart) {
						res.byPL[matchedPL].rejectedRequests++
					}
					retry := s.waitBetweenRequests
					if retry <= 0 {
						retry = 10 * time.Millisecond
					}
					clk.EventAfterDuration(issueNext, retry)
					return
				}

				apfSeats := int64(est.MaxSeats())
				startExecution = func() {
					executed := false
					clk.EventAfterDuration(func(now time.Time) {
						if !executed {
							return
						}
						integrate()
						inFlightAPFSeats -= apfSeats
						inFlightRealSeats -= reqRealSeats
						inFlightRealByPL[matchedPL] -= reqRealSeats
						if matchedFlow != "" {
							inFlightRealByFlow[matchedFlow] -= reqRealSeats
							if !now.Before(measureStart) {
								res.byFlow[matchedFlow].completedRequests++
							}
						}
						if !now.Before(measureStart) {
							res.byPL[matchedPL].completedRequests++
						}
						if now.Before(endTime) {
							if s.waitBetweenRequests <= 0 {
								issueNext(now)
							} else {
								clk.EventAfterDuration(issueNext, s.waitBetweenRequests)
							}
						}
					}, s.requestDuration)

					fqReq.Finish(func() {
						executed = true
					})
					if !executed {
						if !clk.Now().Before(measureStart) {
							res.byPL[matchedPL].rejectedRequests++
						}
						clk.EventAfterDuration(issueNext, 10*time.Millisecond)
						return
					}
					if !clk.Now().Before(measureStart) {
						wait := clk.Now().Sub(arrival)
						res.byPL[matchedPL].queueWaits = append(res.byPL[matchedPL].queueWaits, wait)
						if wait > res.byPL[matchedPL].p99QueueWait {
							res.byPL[matchedPL].p99QueueWait = wait
						}
					}

					integrate()
					inFlightAPFSeats += apfSeats
					inFlightRealSeats += reqRealSeats
					inFlightRealByPL[matchedPL] += reqRealSeats
					if matchedFlow != "" {
						flowReport(matchedFlow)
						inFlightRealByFlow[matchedFlow] += reqRealSeats
					}
				}

				if !inQueueNow {
					startExecution()
				}
			}
			clk.EventAfterDuration(issueNext, 0)
		}
	}

	clk.Run(endTime, verifyQueueSetFidelity)
	integrate()

	evalDuration := endTime.Sub(measureStart)
	if evalDuration > 0 {
		res.avgRealSeats = float64(res.realSeatNanos) / float64(evalDuration)
	}
	for _, st := range res.byPL {
		if evalDuration > 0 {
			st.avgSeats = float64(st.seatNanoseconds) / float64(evalDuration)
		}
		if n := len(st.queueWaits); n > 0 {
			slices.Sort(st.queueWaits)
			st.p99QueueWait = st.queueWaits[int(float64(n-1)*0.99)]
		}
	}
	for _, fr := range res.byFlow {
		if evalDuration > 0 {
			fr.avgSeats = float64(fr.seatNanoseconds) / float64(evalDuration)
		}
	}
	return res
}

// newBootstrapAPFController constructs a real *configController loaded with
// Kubernetes's Mandatory + Suggested PriorityLevelConfigurations and FlowSchemas,
// following the same setup pattern as exempt_borrowing_test.go.
func newBootstrapAPFController(clk eventclock.Interface, serverCL int) *configController {
	k8sClient := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(k8sClient, 0)
	ctlr := newTestableController(TestableConfig{
		Name:                   "sim-apf",
		Clock:                  clk,
		AsFieldManager:         ConfigConsumerAsFieldManager,
		FoundToDangling:        func(found bool) bool { return !found },
		InformerFactory:        informerFactory,
		FlowcontrolClient:      k8sClient.FlowcontrolV1(),
		ServerConcurrencyLimit: serverCL,
		ReqsGaugeVec:           metrics.PriorityLevelConcurrencyGaugeVec,
		ExecSeatsGaugeVec:      metrics.PriorityLevelExecutionSeatsGaugeVec,
		QueueSetFactory:        fqs.NewQueueSetFactory(clk),
	})

	var plcs []*flowcontrol.PriorityLevelConfiguration
	plcs = append(plcs, fcboot.MandatoryPriorityLevelConfigurations...)
	plcs = append(plcs, fcboot.SuggestedPriorityLevelConfigurations...)

	var fses []*flowcontrol.FlowSchema
	fses = append(fses, fcboot.MandatoryFlowSchemas...)
	fses = append(fses, fcboot.SuggestedFlowSchemas...)

	_ = ctlr.lockAndDigestConfigObjects(plcs, fses)
	return ctlr
}

// =============================================================================
// 3. Discrete-Event Clock (Single-Threaded, Zero-Sleep eventclock.Interface)
// =============================================================================

type simClock struct {
	*baseclocktest.FakePassiveClock
	events simEventHeap
	seq    uint64
}

type simEvent struct {
	targetTime time.Time
	seq        uint64
	fn         eventclock.EventFunc
}

type simEventHeap []simEvent

func (h simEventHeap) Len() int { return len(h) }
func (h simEventHeap) Less(i, j int) bool {
	if h[i].targetTime.Equal(h[j].targetTime) {
		return h[i].seq < h[j].seq
	}
	return h[i].targetTime.Before(h[j].targetTime)
}
func (h simEventHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *simEventHeap) Push(x interface{}) { *h = append(*h, x.(simEvent)) }
func (h *simEventHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

func newSimClock(t0 time.Time) *simClock {
	return &simClock{FakePassiveClock: baseclocktest.NewFakePassiveClock(t0)}
}

func (c *simClock) EventAfterDuration(f eventclock.EventFunc, d time.Duration) {
	c.EventAfterTime(f, c.Now().Add(d))
}

func (c *simClock) EventAfterTime(f eventclock.EventFunc, t time.Time) {
	now := c.Now()
	if t.Before(now) {
		t = now
	}
	c.seq++
	heap.Push(&c.events, simEvent{targetTime: t, seq: c.seq, fn: f})
}

func (c *simClock) Sleep(d time.Duration) {
	if d > 0 {
		c.SetTime(c.Now().Add(d))
	}
}

func (c *simClock) Run(limit time.Time, onTimestampSettled func()) {
	for len(c.events) > 0 {
		if c.events[0].targetTime.After(limit) {
			break
		}
		ev := heap.Pop(&c.events).(simEvent)
		if ev.targetTime.After(c.Now()) {
			if onTimestampSettled != nil {
				onTimestampSettled()
			}
			c.SetTime(ev.targetTime)
		}
		ev.fn(c.Now())
	}
	if onTimestampSettled != nil {
		onTimestampSettled()
	}
	if limit.After(c.Now()) {
		c.SetTime(limit)
	}
}
