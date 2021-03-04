/*
Copyright 2015 The Kubernetes Authors.

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

package scheduler

// This file tests scheduler extender.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	filter               = "filter"
	prioritize           = "prioritize"
	bind                 = "bind"
	extendedResourceName = "foo.com/bar"
)

type fitPredicate func(pod *v1.Pod, node *v1.Node) (bool, error)
type priorityFunc func(pod *v1.Pod, nodes *v1.NodeList) (*extenderv1.HostPriorityList, error)

type priorityConfig struct {
	function priorityFunc
	weight   int64
}

type Extender struct {
	name             string
	predicates       []fitPredicate
	prioritizers     []priorityConfig
	nodeCacheCapable bool
	Client           clientset.Interface
}

func (e *Extender) serveHTTP(t *testing.T, w http.ResponseWriter, req *http.Request) {
	decoder := json.NewDecoder(req.Body)
	defer req.Body.Close()

	encoder := json.NewEncoder(w)

	if strings.Contains(req.URL.Path, filter) || strings.Contains(req.URL.Path, prioritize) {
		var args extenderv1.ExtenderArgs

		if err := decoder.Decode(&args); err != nil {
			http.Error(w, "Decode error", http.StatusBadRequest)
			return
		}

		if strings.Contains(req.URL.Path, filter) {
			resp, err := e.Filter(&args)
			if err != nil {
				resp.Error = err.Error()
			}

			if err := encoder.Encode(resp); err != nil {
				t.Fatalf("Failed to encode %v", resp)
			}
		} else if strings.Contains(req.URL.Path, prioritize) {
			// Prioritize errors are ignored. Default k8s priorities or another extender's
			// priorities may be applied.
			priorities, _ := e.Prioritize(&args)

			if err := encoder.Encode(priorities); err != nil {
				t.Fatalf("Failed to encode %+v", priorities)
			}
		}
	} else if strings.Contains(req.URL.Path, bind) {
		var args extenderv1.ExtenderBindingArgs

		if err := decoder.Decode(&args); err != nil {
			http.Error(w, "Decode error", http.StatusBadRequest)
			return
		}

		resp := &extenderv1.ExtenderBindingResult{}

		if err := e.Bind(&args); err != nil {
			resp.Error = err.Error()
		}

		if err := encoder.Encode(resp); err != nil {
			t.Fatalf("Failed to encode %+v", resp)
		}
	} else {
		http.Error(w, "Unknown method", http.StatusNotFound)
	}
}

func (e *Extender) filterUsingNodeCache(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
	nodeSlice := make([]string, 0)
	failedNodesMap := extenderv1.FailedNodesMap{}
	for _, nodeName := range *args.NodeNames {
		fits := true
		for _, predicate := range e.predicates {
			fit, err := predicate(args.Pod,
				&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
			if err != nil {
				return &extenderv1.ExtenderFilterResult{
					Nodes:       nil,
					NodeNames:   nil,
					FailedNodes: extenderv1.FailedNodesMap{},
					Error:       err.Error(),
				}, err
			}
			if !fit {
				fits = false
				break
			}
		}
		if fits {
			nodeSlice = append(nodeSlice, nodeName)
		} else {
			failedNodesMap[nodeName] = fmt.Sprintf("extender failed: %s", e.name)
		}
	}

	return &extenderv1.ExtenderFilterResult{
		Nodes:       nil,
		NodeNames:   &nodeSlice,
		FailedNodes: failedNodesMap,
	}, nil
}

func (e *Extender) Filter(args *extenderv1.ExtenderArgs) (*extenderv1.ExtenderFilterResult, error) {
	filtered := []v1.Node{}
	failedNodesMap := extenderv1.FailedNodesMap{}

	if e.nodeCacheCapable {
		return e.filterUsingNodeCache(args)
	}

	for _, node := range args.Nodes.Items {
		fits := true
		for _, predicate := range e.predicates {
			fit, err := predicate(args.Pod, &node)
			if err != nil {
				return &extenderv1.ExtenderFilterResult{
					Nodes:       &v1.NodeList{},
					NodeNames:   nil,
					FailedNodes: extenderv1.FailedNodesMap{},
					Error:       err.Error(),
				}, err
			}
			if !fit {
				fits = false
				break
			}
		}
		if fits {
			filtered = append(filtered, node)
		} else {
			failedNodesMap[node.Name] = fmt.Sprintf("extender failed: %s", e.name)
		}
	}

	return &extenderv1.ExtenderFilterResult{
		Nodes:       &v1.NodeList{Items: filtered},
		NodeNames:   nil,
		FailedNodes: failedNodesMap,
	}, nil
}

func (e *Extender) Prioritize(args *extenderv1.ExtenderArgs) (*extenderv1.HostPriorityList, error) {
	result := extenderv1.HostPriorityList{}
	combinedScores := map[string]int64{}
	var nodes = &v1.NodeList{Items: []v1.Node{}}

	if e.nodeCacheCapable {
		for _, nodeName := range *args.NodeNames {
			nodes.Items = append(nodes.Items, v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
		}
	} else {
		nodes = args.Nodes
	}

	for _, prioritizer := range e.prioritizers {
		weight := prioritizer.weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.function
		prioritizedList, err := priorityFunc(args.Pod, nodes)
		if err != nil {
			return &extenderv1.HostPriorityList{}, err
		}
		for _, hostEntry := range *prioritizedList {
			combinedScores[hostEntry.Host] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		result = append(result, extenderv1.HostPriority{Host: host, Score: score})
	}
	return &result, nil
}

func (e *Extender) Bind(binding *extenderv1.ExtenderBindingArgs) error {
	b := &v1.Binding{
		ObjectMeta: metav1.ObjectMeta{Namespace: binding.PodNamespace, Name: binding.PodName, UID: binding.PodUID},
		Target: v1.ObjectReference{
			Kind: "Node",
			Name: binding.Node,
		},
	}

	return e.Client.CoreV1().Pods(b.Namespace).Bind(context.TODO(), b, metav1.CreateOptions{})
}

func machine1_2_3Predicate(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "machine1" || node.Name == "machine2" || node.Name == "machine3" {
		return true, nil
	}
	return false, nil
}

func machine2_3_5Predicate(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "machine2" || node.Name == "machine3" || node.Name == "machine5" {
		return true, nil
	}
	return false, nil
}

func machine2Prioritizer(pod *v1.Pod, nodes *v1.NodeList) (*extenderv1.HostPriorityList, error) {
	result := extenderv1.HostPriorityList{}
	for _, node := range nodes.Items {
		score := 1
		if node.Name == "machine2" {
			score = 10
		}
		result = append(result, extenderv1.HostPriority{
			Host:  node.Name,
			Score: int64(score),
		})
	}
	return &result, nil
}

func machine3Prioritizer(pod *v1.Pod, nodes *v1.NodeList) (*extenderv1.HostPriorityList, error) {
	result := extenderv1.HostPriorityList{}
	for _, node := range nodes.Items {
		score := 1
		if node.Name == "machine3" {
			score = 10
		}
		result = append(result, extenderv1.HostPriority{
			Host:  node.Name,
			Score: int64(score),
		})
	}
	return &result, nil
}

func TestSchedulerExtender(t *testing.T) {
	testCtx := testutils.InitTestMaster(t, "scheduler-extender", nil)
	clientSet := testCtx.ClientSet

	extender1 := &Extender{
		name:         "extender1",
		predicates:   []fitPredicate{machine1_2_3Predicate},
		prioritizers: []priorityConfig{{machine2Prioritizer, 1}},
	}
	es1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		extender1.serveHTTP(t, w, req)
	}))
	defer es1.Close()

	extender2 := &Extender{
		name:         "extender2",
		predicates:   []fitPredicate{machine2_3_5Predicate},
		prioritizers: []priorityConfig{{machine3Prioritizer, 1}},
		Client:       clientSet,
	}
	es2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		extender2.serveHTTP(t, w, req)
	}))
	defer es2.Close()

	extender3 := &Extender{
		name:             "extender3",
		predicates:       []fitPredicate{machine1_2_3Predicate},
		prioritizers:     []priorityConfig{{machine2Prioritizer, 5}},
		nodeCacheCapable: true,
	}
	es3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		extender3.serveHTTP(t, w, req)
	}))
	defer es3.Close()

	policy := schedulerapi.Policy{
		Extenders: []schedulerapi.Extender{
			{
				URLPrefix:      es1.URL,
				FilterVerb:     filter,
				PrioritizeVerb: prioritize,
				Weight:         3,
				EnableHTTPS:    false,
			},
			{
				URLPrefix:      es2.URL,
				FilterVerb:     filter,
				PrioritizeVerb: prioritize,
				BindVerb:       bind,
				Weight:         4,
				EnableHTTPS:    false,
				ManagedResources: []schedulerapi.ExtenderManagedResource{
					{
						Name:               extendedResourceName,
						IgnoredByScheduler: true,
					},
				},
			},
			{
				URLPrefix:        es3.URL,
				FilterVerb:       filter,
				PrioritizeVerb:   prioritize,
				Weight:           10,
				EnableHTTPS:      false,
				NodeCacheCapable: true,
			},
		},
	}
	policy.APIVersion = "v1"

	testCtx = testutils.InitTestScheduler(t, testCtx, &policy)
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	defer testutils.CleanupTest(t, testCtx)

	DoTestPodScheduling(testCtx.NS, t, clientSet)
}

func DoTestPodScheduling(ns *v1.Namespace, t *testing.T, cs clientset.Interface) {
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer cs.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})

	goodCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: metav1.Time{Time: time.Now()},
	}
	node := &v1.Node{
		Spec: v1.NodeSpec{Unschedulable: false},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
			Conditions: []v1.NodeCondition{goodCondition},
		},
	}

	for ii := 0; ii < 5; ii++ {
		node.Name = fmt.Sprintf("machine%d", ii+1)
		if _, err := createNode(cs, node); err != nil {
			t.Fatalf("Failed to create nodes: %v", err)
		}
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "extender-test-pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							extendedResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	}

	myPod, err := cs.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	err = wait.Poll(time.Second, wait.ForeverTestTimeout, testutils.PodScheduled(cs, myPod.Namespace, myPod.Name))
	if err != nil {
		t.Fatalf("Failed to schedule pod: %v", err)
	}

	myPod, err = cs.CoreV1().Pods(ns.Name).Get(context.TODO(), myPod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod: %v", err)
	} else if myPod.Spec.NodeName != "machine2" {
		t.Fatalf("Failed to schedule using extender, expected machine2, got %v", myPod.Spec.NodeName)
	}
	var gracePeriod int64
	if err := cs.CoreV1().Pods(ns.Name).Delete(context.TODO(), myPod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}); err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	_, err = cs.CoreV1().Pods(ns.Name).Get(context.TODO(), myPod.Name, metav1.GetOptions{})
	if err == nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	t.Logf("Scheduled pod using extenders")
}
