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

package preemption

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

const (
	extenderFilterVerb    = "filter"
	extenderPreemptVerb   = "preempt"
	customGPUResource     = "example.com/gpu"
	customLicenseResource = "example.com/license"

	lowPriority  = 10
	midPriority  = 100
	highPriority = 1000
)

type fakeExtenderHandler struct {
	mu                sync.Mutex
	filterFunc        func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error)
	preemptFunc       func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error)
	filterCalls       int32
	preemptCalls      int32
	filterDelay       time.Duration
	preemptDelay      time.Duration
	filterStatusCode  int
	preemptStatusCode int
}

func newFakeExtenderServer(t *testing.T, handler *fakeExtenderHandler) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		decoder := json.NewDecoder(req.Body)
		defer req.Body.Close()
		encoder := json.NewEncoder(w)

		switch {
		case strings.Contains(req.URL.Path, extenderFilterVerb):
			atomic.AddInt32(&handler.filterCalls, 1)
			if handler.filterDelay > 0 {
				time.Sleep(handler.filterDelay)
			}
			handler.mu.Lock()
			code := handler.filterStatusCode
			f := handler.filterFunc
			handler.mu.Unlock()

			if code != 0 && code != http.StatusOK {
				http.Error(w, fmt.Sprintf("simulated filter error %d", code), code)
				return
			}

			var args extenderv1.ExtenderArgs
			if err := decoder.Decode(&args); err != nil {
				http.Error(w, fmt.Sprintf("decode error: %v", err), http.StatusBadRequest)
				return
			}

			if f != nil {
				res, err := f(&args)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				_ = encoder.Encode(res)
			} else {
				// Default filter: all nodes pass
				res := &extenderv1.ExtenderFilterResult{}
				if args.Nodes != nil {
					res.Nodes = args.Nodes
				}
				if args.NodeNames != nil {
					res.NodeNames = args.NodeNames
				}
				_ = encoder.Encode(res)
			}

		case strings.Contains(req.URL.Path, extenderPreemptVerb):
			atomic.AddInt32(&handler.preemptCalls, 1)
			if handler.preemptDelay > 0 {
				time.Sleep(handler.preemptDelay)
			}
			handler.mu.Lock()
			code := handler.preemptStatusCode
			f := handler.preemptFunc
			handler.mu.Unlock()

			if code != 0 && code != http.StatusOK {
				http.Error(w, fmt.Sprintf("simulated preempt error %d", code), code)
				return
			}

			var args extenderv1.ExtenderPreemptionArgs
			if err := decoder.Decode(&args); err != nil {
				http.Error(w, fmt.Sprintf("decode error: %v", err), http.StatusBadRequest)
				return
			}

			if f != nil {
				res, err := f(&args)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				_ = encoder.Encode(res)
			} else {
				// Default preempt: return empty meta victims
				res := &extenderv1.ExtenderPreemptionResult{
					NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{},
				}
				_ = encoder.Encode(res)
			}

		default:
			http.Error(w, fmt.Sprintf("unknown verb %s", req.URL.Path), http.StatusNotFound)
		}
	}))
}

// createTestNodeWithResources creates a ready node with specified allocatable resources.
func createTestNodeWithResources(cs clientset.Interface, name string, cpuMillis int64, memoryMB int64) (*v1.Node, error) {
	node := st.MakeNode().Name(name).
		Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "32",
			v1.ResourceCPU:    fmt.Sprintf("%dm", cpuMillis),
			v1.ResourceMemory: fmt.Sprintf("%dMi", memoryMB),
		}).
		Obj()
	return testutils.CreateNode(cs, node)
}

func makePodWithResources(ns, name, nodeName string, priority int32, reqMap map[v1.ResourceName]string) *v1.Pod {
	pw := st.MakePod().Namespace(ns).Name(name).
		Priority(priority).
		Res(reqMap).
		ZeroTerminationGracePeriod()
	if nodeName != "" {
		pw.Node(nodeName)
	}
	return pw.Obj()
}

// simulateVictimDeletion watches for the victim pod getting evicted, then deletes it with gracePeriod=0
// to simulate kubelet finishing pod termination.
func simulateVictimDeletion(ctx context.Context, cs clientset.Interface, ns, name string) error {
	err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		return pod.DeletionTimestamp != nil, nil
	})
	if err != nil {
		return fmt.Errorf("timed out waiting for victim pod %s/%s eviction: %w", ns, name, err)
	}
	var zero int64
	err = cs.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{GracePeriodSeconds: &zero})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// TestExtenderPreemption_EmptyInTreeVictimsPlaceholder verifies FM-204 / PR #135486:
// When in-tree filter plugins fit without preempting any pods (empty in-tree victims),
// the candidate node placeholder is preserved and passed to preempt-capable extenders.
// The extender nominates victim pods holding custom resources, and the preemptor is scheduled
// after victim eviction.
func TestExtenderPreemption_EmptyInTreeVictimsPlaceholder(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "ext-empty", nil)
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	nodeName := "node-empty-victims"
	// Node has plenty of CPU/Memory (in-tree filters fit without victims)
	if _, err := createTestNodeWithResources(cs, nodeName, 8000, 8192); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create a low-priority victim pod running on the node holding the custom GPU resource
	victimPod := makePodWithResources(ns, "victim-gpu-pod", nodeName, midPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100Mi",
		customGPUResource: "1",
	})

	victimPod, err := runPausePod(cs, victimPod)
	if err != nil {
		t.Fatalf("Failed to run victim pod: %v", err)
	}

	var observedEmptyVictimsOnPreempt atomic.Bool

	extHandler := &fakeExtenderHandler{
		filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
			// Check if victim pod is still alive
			_, err := cs.CoreV1().Pods(ns).Get(context.Background(), victimPod.Name, metav1.GetOptions{})
			if err == nil {
				// Victim using GPU, reject node
				res := &extenderv1.ExtenderFilterResult{
					Nodes:       &v1.NodeList{},
					FailedNodes: extenderv1.FailedNodesMap{nodeName: "out of custom GPU resources"},
				}
				return res, nil
			}
			// Victim deleted, node has capacity
			res := &extenderv1.ExtenderFilterResult{
				Nodes: args.Nodes,
			}
			return res, nil
		},
		preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
			// Invariant check: In-tree preemption must pass empty victims placeholder for nodeName!
			if args.NodeNameToVictims != nil {
				victims, ok := args.NodeNameToVictims[nodeName]
				if ok && (victims == nil || len(victims.Pods) == 0) {
					observedEmptyVictimsOnPreempt.Store(true)
				}
			}
			if args.NodeNameToMetaVictims != nil {
				victims, ok := args.NodeNameToMetaVictims[nodeName]
				if ok && (victims == nil || len(victims.Pods) == 0) {
					observedEmptyVictimsOnPreempt.Store(true)
				}
			}

			// Extender identifies victimPod as the holder of the custom GPU resource
			return &extenderv1.ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
					nodeName: {
						Pods: []*extenderv1.MetaPod{{UID: string(victimPod.UID)}},
					},
				},
			}, nil
		},
	}

	server := newFakeExtenderServer(t, extHandler)
	defer server.Close()

	extenders := []schedulerapi.Extender{
		{
			URLPrefix:   server.URL,
			FilterVerb:  extenderFilterVerb,
			PreemptVerb: extenderPreemptVerb,
			EnableHTTPS: false,
			ManagedResources: []schedulerapi.ExtenderManagedResource{
				{
					Name:               customGPUResource,
					IgnoredByScheduler: true,
				},
			},
			Ignorable: false,
		},
	}

	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithExtenders(extenders...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	// Create high-priority preemptor requesting the custom GPU resource
	preemptorPod := makePodWithResources(ns, "preemptor-gpu-pod", "", highPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100Mi",
		customGPUResource: "1",
	})

	preemptorPod, err = createPausePod(cs, preemptorPod)
	if err != nil {
		t.Fatalf("Failed to create preemptor pod: %v", err)
	}

	// Verify victim pod gets evicted and deleted
	if err := simulateVictimDeletion(testCtx.Ctx, cs, ns, victimPod.Name); err != nil {
		t.Fatalf("Error simulating victim deletion: %v", err)
	}

	// Verify preemptor pod is successfully scheduled onto node-empty-victims
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
		testutils.PodScheduled(cs, ns, preemptorPod.Name))
	if err != nil {
		t.Fatalf("Preemptor pod failed to schedule: %v", err)
	}

	scheduledPod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, preemptorPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get scheduled preemptor pod: %v", err)
	}
	if scheduledPod.Spec.NodeName != nodeName {
		t.Fatalf("Expected preemptor to be scheduled on %s, got %s", nodeName, scheduledPod.Spec.NodeName)
	}

	if !observedEmptyVictimsOnPreempt.Load() {
		t.Fatalf("Expected extender ProcessPreemption to receive placeholder empty-victim candidate for node %s", nodeName)
	}
	if atomic.LoadInt32(&extHandler.preemptCalls) == 0 {
		t.Fatalf("Expected extender ProcessPreemption to be called at least once")
	}
}

// TestExtenderPreemption_CombinedInTreeAndExtenderVictims verifies that candidate nomination
// and victim eviction correctly combine in-tree victims (e.g. CPU) and extender victims (custom resource).
func TestExtenderPreemption_CombinedInTreeAndExtenderVictims(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "ext-comb", nil)
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	nodeName := "node-combined-victims"
	// Node has 1000m CPU total capacity
	if _, err := createTestNodeWithResources(cs, nodeName, 1000, 4096); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Victim 1: Consumes 800m CPU (in-tree resource conflict)
	victimCPU := makePodWithResources(ns, "victim-cpu-pod", nodeName, midPriority, map[v1.ResourceName]string{
		v1.ResourceCPU: "800m",
	})
	victimCPU, err := runPausePod(cs, victimCPU)
	if err != nil {
		t.Fatalf("Failed to run victim CPU pod: %v", err)
	}

	// Victim 2: Consumes 100m CPU + 1 custom GPU (extender resource conflict)
	victimGPU := makePodWithResources(ns, "victim-gpu-pod", nodeName, midPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		customGPUResource: "1",
	})
	victimGPU, err = runPausePod(cs, victimGPU)
	if err != nil {
		t.Fatalf("Failed to run victim GPU pod: %v", err)
	}

	var observedInTreeVictimOnPreempt atomic.Bool

	extHandler := &fakeExtenderHandler{
		filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
			_, err := cs.CoreV1().Pods(ns).Get(context.Background(), victimGPU.Name, metav1.GetOptions{})
			if err == nil {
				res := &extenderv1.ExtenderFilterResult{
					Nodes:       &v1.NodeList{},
					FailedNodes: extenderv1.FailedNodesMap{nodeName: "out of custom GPU"},
				}
				return res, nil
			}
			res := &extenderv1.ExtenderFilterResult{
				Nodes: args.Nodes,
			}
			return res, nil
		},
		preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
			// In-tree preemption should already have identified victimCPU for nodeName!
			var inTreePods []*v1.Pod
			if args.NodeNameToVictims != nil && args.NodeNameToVictims[nodeName] != nil {
				inTreePods = args.NodeNameToVictims[nodeName].Pods
			}
			for _, p := range inTreePods {
				if p.UID == victimCPU.UID {
					observedInTreeVictimOnPreempt.Store(true)
				}
			}

			// Extender returns combined victims: victimCPU + victimGPU
			return &extenderv1.ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
					nodeName: {
						Pods: []*extenderv1.MetaPod{
							{UID: string(victimCPU.UID)},
							{UID: string(victimGPU.UID)},
						},
					},
				},
			}, nil
		},
	}

	server := newFakeExtenderServer(t, extHandler)
	defer server.Close()

	extenders := []schedulerapi.Extender{
		{
			URLPrefix:   server.URL,
			FilterVerb:  extenderFilterVerb,
			PreemptVerb: extenderPreemptVerb,
			EnableHTTPS: false,
			ManagedResources: []schedulerapi.ExtenderManagedResource{
				{
					Name:               customGPUResource,
					IgnoredByScheduler: true,
				},
			},
			Ignorable: false,
		},
	}

	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithExtenders(extenders...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	// Preemptor needs 500m CPU (conflicts with victimCPU) and 1 GPU (conflicts with victimGPU)
	preemptorPod := makePodWithResources(ns, "preemptor-combined-pod", "", highPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "500m",
		customGPUResource: "1",
	})

	preemptorPod, err = createPausePod(cs, preemptorPod)
	if err != nil {
		t.Fatalf("Failed to create preemptor pod: %v", err)
	}

	// Verify both victims get evicted and deleted
	if err := simulateVictimDeletion(testCtx.Ctx, cs, ns, victimCPU.Name); err != nil {
		t.Fatalf("Error simulating victimCPU deletion: %v", err)
	}
	if err := simulateVictimDeletion(testCtx.Ctx, cs, ns, victimGPU.Name); err != nil {
		t.Fatalf("Error simulating victimGPU deletion: %v", err)
	}

	// Verify preemptor is scheduled
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
		testutils.PodScheduled(cs, ns, preemptorPod.Name))
	if err != nil {
		t.Fatalf("Preemptor pod failed to schedule: %v", err)
	}

	if !observedInTreeVictimOnPreempt.Load() {
		t.Fatalf("Expected in-tree victimCPU to be passed to extender ProcessPreemption")
	}
}

// TestExtenderPreemption_ExtenderFailuresAndTimeout tests failure edge cases:
// 1. Non-ignorable extender returns 500 on preempt: preemption fails cleanly without scheduler panic or invalid node nomination.
// 2. Ignorable extender returns 500 on preempt: placeholder empty-victim node is dropped without panic, preemptor remains unscheduled.
// 3. Extender times out on preempt: handled cleanly based on Ignorable setting without scheduler panic.
// 4. Extender omits node / returns empty victims: placeholder empty-victim node is cleanly discarded.
func TestExtenderPreemption_ExtenderFailuresAndTimeout(t *testing.T) {
	tests := []struct {
		name              string
		ignorable         bool
		preemptStatusCode int
		preemptDelay      time.Duration
		timeout           time.Duration
		omitNode          bool
		returnEmpty       bool
	}{
		{
			name:              "Non-ignorable extender returns HTTP 500 error on preempt",
			ignorable:         false,
			preemptStatusCode: http.StatusInternalServerError,
		},
		{
			name:              "Ignorable extender returns HTTP 500 error on preempt",
			ignorable:         true,
			preemptStatusCode: http.StatusInternalServerError,
		},
		{
			name:         "Non-ignorable extender times out on preempt",
			ignorable:    false,
			preemptDelay: 500 * time.Millisecond,
			timeout:      100 * time.Millisecond,
		},
		{
			name:         "Ignorable extender times out on preempt",
			ignorable:    true,
			preemptDelay: 500 * time.Millisecond,
			timeout:      100 * time.Millisecond,
		},
		{
			name:      "Extender omits candidate node from preemption result",
			ignorable: false,
			omitNode:  true,
		},
		{
			name:        "Extender returns empty victim list for placeholder candidate",
			ignorable:   false,
			returnEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestAPIServer(t, "ext-fail", nil)
			cs := testCtx.ClientSet
			ns := testCtx.NS.Name

			nodeName := "node-failure-test"
			if _, err := createTestNodeWithResources(cs, nodeName, 8000, 8192); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}

			victimPod := makePodWithResources(ns, "victim-failure-test", nodeName, midPriority, map[v1.ResourceName]string{
				customGPUResource: "1",
			})
			victimPod, err := runPausePod(cs, victimPod)
			if err != nil {
				t.Fatalf("Failed to run victim pod: %v", err)
			}

			extHandler := &fakeExtenderHandler{
				preemptStatusCode: tt.preemptStatusCode,
				preemptDelay:      tt.preemptDelay,
				filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
					return &extenderv1.ExtenderFilterResult{
						Nodes:       &v1.NodeList{},
						FailedNodes: extenderv1.FailedNodesMap{nodeName: "extender filter rejected"},
					}, nil
				},
				preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
					if tt.omitNode {
						// Omit node completely to reject preemption on this node
						return &extenderv1.ExtenderPreemptionResult{
							NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{},
						}, nil
					}
					if tt.returnEmpty {
						// Return empty victims
						return &extenderv1.ExtenderPreemptionResult{
							NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
								nodeName: {Pods: []*extenderv1.MetaPod{}},
							},
						}, nil
					}
					return &extenderv1.ExtenderPreemptionResult{
						NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
							nodeName: {
								Pods: []*extenderv1.MetaPod{{UID: string(victimPod.UID)}},
							},
						},
					}, nil
				},
			}

			server := newFakeExtenderServer(t, extHandler)
			defer server.Close()

			extenderConfig := schedulerapi.Extender{
				URLPrefix:   server.URL,
				FilterVerb:  extenderFilterVerb,
				PreemptVerb: extenderPreemptVerb,
				EnableHTTPS: false,
				ManagedResources: []schedulerapi.ExtenderManagedResource{
					{
						Name:               customGPUResource,
						IgnoredByScheduler: true,
					},
				},
				Ignorable: tt.ignorable,
			}
			if tt.timeout > 0 {
				extenderConfig.HTTPTimeout = metav1.Duration{Duration: tt.timeout}
			}

			testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithExtenders(extenderConfig))
			testutils.SyncSchedulerInformerFactory(testCtx)
			go testCtx.Scheduler.Run(testCtx.Ctx)

			preemptorPod := makePodWithResources(ns, "preemptor-failure-test", "", highPriority, map[v1.ResourceName]string{
				customGPUResource: "1",
			})

			preemptorPod, err = createPausePod(cs, preemptorPod)
			if err != nil {
				t.Fatalf("Failed to create preemptor pod: %v", err)
			}

			// Verify that the preemptor is NOT scheduled and victim is NOT evicted
			time.Sleep(1 * time.Second)

			currentVictim, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, victimPod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get victim pod: %v", err)
			}
			if currentVictim.DeletionTimestamp != nil {
				t.Fatalf("Victim pod %s was unexpectedly evicted/deleted", victimPod.Name)
			}

			currentPreemptor, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, preemptorPod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to get preemptor pod: %v", err)
			}
			if currentPreemptor.Spec.NodeName != "" {
				t.Fatalf("Preemptor pod was unexpectedly scheduled to %s", currentPreemptor.Spec.NodeName)
			}
		})
	}
}

// TestExtenderPreemption_ChainedExtendersPassthrough verifies that when multiple extenders are chained,
// an extender can leave an empty-victim placeholder node unchanged (or passthrough), and a subsequent
// extender can add victims for its managed resource.
func TestExtenderPreemption_ChainedExtendersPassthrough(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "ext-chain", nil)
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	nodeName := "node-chained-extenders"
	if _, err := createTestNodeWithResources(cs, nodeName, 8000, 8192); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Victim pod holds custom license resource (managed by Extender B)
	victimPod := makePodWithResources(ns, "victim-license-pod", nodeName, midPriority, map[v1.ResourceName]string{
		customLicenseResource: "1",
	})
	victimPod, err := runPausePod(cs, victimPod)
	if err != nil {
		t.Fatalf("Failed to run victim pod: %v", err)
	}

	var extenderAObservedPlaceholder atomic.Bool
	var extenderBObservedPlaceholder atomic.Bool

	// Extender A: Manages customGPUResource. Preemptor requests both GPU and License.
	// Extender A has enough GPU capacity, so it leaves placeholder unchanged (empty victims).
	extHandlerA := &fakeExtenderHandler{
		filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
			// Extender A allows node
			res := &extenderv1.ExtenderFilterResult{}
			if args.Nodes != nil {
				res.Nodes = args.Nodes
			}
			return res, nil
		},
		preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
			if args.NodeNameToVictims != nil && args.NodeNameToVictims[nodeName] != nil {
				if len(args.NodeNameToVictims[nodeName].Pods) == 0 {
					extenderAObservedPlaceholder.Store(true)
				}
			}
			// Keep placeholder unchanged for downstream extenders: return empty victims map entry for nodeName
			return &extenderv1.ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
					nodeName: {Pods: []*extenderv1.MetaPod{}},
				},
			}, nil
		},
	}
	serverA := newFakeExtenderServer(t, extHandlerA)
	defer serverA.Close()

	// Extender B: Manages customLicenseResource. Rejects node in filter and adds victimPod in preempt.
	extHandlerB := &fakeExtenderHandler{
		filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
			_, err := cs.CoreV1().Pods(ns).Get(context.Background(), victimPod.Name, metav1.GetOptions{})
			if err == nil {
				return &extenderv1.ExtenderFilterResult{
					Nodes:       &v1.NodeList{},
					FailedNodes: extenderv1.FailedNodesMap{nodeName: "out of custom licenses"},
				}, nil
			}
			return &extenderv1.ExtenderFilterResult{
				Nodes: args.Nodes,
			}, nil
		},
		preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
			if args.NodeNameToVictims != nil && args.NodeNameToVictims[nodeName] != nil {
				if len(args.NodeNameToVictims[nodeName].Pods) == 0 {
					extenderBObservedPlaceholder.Store(true)
				}
			}
			return &extenderv1.ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
					nodeName: {
						Pods: []*extenderv1.MetaPod{{UID: string(victimPod.UID)}},
					},
				},
			}, nil
		},
	}
	serverB := newFakeExtenderServer(t, extHandlerB)
	defer serverB.Close()

	extenders := []schedulerapi.Extender{
		{
			URLPrefix:   serverA.URL,
			FilterVerb:  extenderFilterVerb,
			PreemptVerb: extenderPreemptVerb,
			EnableHTTPS: false,
			ManagedResources: []schedulerapi.ExtenderManagedResource{
				{Name: customGPUResource, IgnoredByScheduler: true},
			},
			Ignorable: false,
		},
		{
			URLPrefix:   serverB.URL,
			FilterVerb:  extenderFilterVerb,
			PreemptVerb: extenderPreemptVerb,
			EnableHTTPS: false,
			ManagedResources: []schedulerapi.ExtenderManagedResource{
				{Name: customLicenseResource, IgnoredByScheduler: true},
			},
			Ignorable: false,
		},
	}

	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithExtenders(extenders...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	preemptorPod := makePodWithResources(ns, "preemptor-chained-pod", "", highPriority, map[v1.ResourceName]string{
		customGPUResource:     "1",
		customLicenseResource: "1",
	})

	preemptorPod, err = createPausePod(cs, preemptorPod)
	if err != nil {
		t.Fatalf("Failed to create preemptor pod: %v", err)
	}

	// Verify victim pod gets evicted and deleted
	if err := simulateVictimDeletion(testCtx.Ctx, cs, ns, victimPod.Name); err != nil {
		t.Fatalf("Error simulating victim deletion: %v", err)
	}

	// Verify preemptor pod is scheduled onto node-chained-extenders
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
		testutils.PodScheduled(cs, ns, preemptorPod.Name))
	if err != nil {
		t.Fatalf("Preemptor pod failed to schedule: %v", err)
	}

	if !extenderAObservedPlaceholder.Load() {
		t.Fatalf("Expected Extender A to observe empty victim placeholder")
	}
	if !extenderBObservedPlaceholder.Load() {
		t.Fatalf("Expected Extender B to observe empty victim placeholder passed through Extender A")
	}
}

// TestExtenderPreemption_NodeCacheCapable verifies preemption behavior when the extender
// has NodeCacheCapable enabled (uses MetaVictims with cached node info).
func TestExtenderPreemption_NodeCacheCapable(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "ext-cache", nil)
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	nodeName := "node-cache-capable"
	if _, err := createTestNodeWithResources(cs, nodeName, 8000, 8192); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	victimPod := makePodWithResources(ns, "victim-cache-pod", nodeName, midPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100Mi",
		customGPUResource: "1",
	})

	victimPod, err := runPausePod(cs, victimPod)
	if err != nil {
		t.Fatalf("Failed to run victim pod: %v", err)
	}

	var observedMetaVictimsPlaceholder atomic.Bool

	extHandler := &fakeExtenderHandler{
		filterFunc: func(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
			_, err := cs.CoreV1().Pods(ns).Get(context.Background(), victimPod.Name, metav1.GetOptions{})
			if err == nil {
				// With NodeCacheCapable, args.NodeNames is provided
				res := &extenderv1.ExtenderFilterResult{
					NodeNames:   &[]string{},
					FailedNodes: extenderv1.FailedNodesMap{nodeName: "out of custom GPU"},
				}
				return res, nil
			}
			res := &extenderv1.ExtenderFilterResult{
				NodeNames: args.NodeNames,
			}
			return res, nil
		},
		preemptFunc: func(args *extenderv1.ExtenderPreemptionArgs) (*extenderv1.ExtenderPreemptionResult, error) {
			// With NodeCacheCapable, NodeNameToMetaVictims is passed
			if args.NodeNameToMetaVictims != nil {
				victims, ok := args.NodeNameToMetaVictims[nodeName]
				if ok && (victims == nil || len(victims.Pods) == 0) {
					observedMetaVictimsPlaceholder.Store(true)
				}
			}

			return &extenderv1.ExtenderPreemptionResult{
				NodeNameToMetaVictims: map[string]*extenderv1.MetaVictims{
					nodeName: {
						Pods: []*extenderv1.MetaPod{{UID: string(victimPod.UID)}},
					},
				},
			}, nil
		},
	}

	server := newFakeExtenderServer(t, extHandler)
	defer server.Close()

	extenders := []schedulerapi.Extender{
		{
			URLPrefix:        server.URL,
			FilterVerb:       extenderFilterVerb,
			PreemptVerb:      extenderPreemptVerb,
			EnableHTTPS:      false,
			NodeCacheCapable: true,
			ManagedResources: []schedulerapi.ExtenderManagedResource{
				{
					Name:               customGPUResource,
					IgnoredByScheduler: true,
				},
			},
			Ignorable: false,
		},
	}

	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, scheduler.WithExtenders(extenders...))
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	preemptorPod := makePodWithResources(ns, "preemptor-cache-pod", "", highPriority, map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100Mi",
		customGPUResource: "1",
	})

	preemptorPod, err = createPausePod(cs, preemptorPod)
	if err != nil {
		t.Fatalf("Failed to create preemptor pod: %v", err)
	}

	if err := simulateVictimDeletion(testCtx.Ctx, cs, ns, victimPod.Name); err != nil {
		t.Fatalf("Error simulating victim deletion: %v", err)
	}

	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
		testutils.PodScheduled(cs, ns, preemptorPod.Name))
	if err != nil {
		t.Fatalf("Preemptor pod failed to schedule: %v", err)
	}

	if !observedMetaVictimsPlaceholder.Load() {
		t.Fatalf("Expected extender ProcessPreemption to receive NodeNameToMetaVictims placeholder for node %s", nodeName)
	}
}
