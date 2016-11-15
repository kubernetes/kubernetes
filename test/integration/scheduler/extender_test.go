// +build integration,!no-etcd

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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	filter     = "filter"
	prioritize = "prioritize"
)

type fitPredicate func(pod *api.Pod, node *api.Node) (bool, error)
type priorityFunc func(pod *api.Pod, nodes *api.NodeList) (*schedulerapi.HostPriorityList, error)

type priorityConfig struct {
	function priorityFunc
	weight   int
}

type Extender struct {
	name         string
	predicates   []fitPredicate
	prioritizers []priorityConfig
}

func (e *Extender) serveHTTP(t *testing.T, w http.ResponseWriter, req *http.Request) {
	var args schedulerapi.ExtenderArgs

	decoder := json.NewDecoder(req.Body)
	defer req.Body.Close()

	if err := decoder.Decode(&args); err != nil {
		http.Error(w, "Decode error", http.StatusBadRequest)
		return
	}

	encoder := json.NewEncoder(w)

	if strings.Contains(req.URL.Path, filter) {
		resp := &schedulerapi.ExtenderFilterResult{}
		nodes, failedNodes, err := e.Filter(&args.Pod, &args.Nodes)
		if err != nil {
			resp.Error = err.Error()
		} else {
			resp.Nodes = *nodes
			resp.FailedNodes = failedNodes
		}

		if err := encoder.Encode(resp); err != nil {
			t.Fatalf("Failed to encode %+v", resp)
		}
	} else if strings.Contains(req.URL.Path, prioritize) {
		// Prioritize errors are ignored. Default k8s priorities or another extender's
		// priorities may be applied.
		priorities, _ := e.Prioritize(&args.Pod, &args.Nodes)

		if err := encoder.Encode(priorities); err != nil {
			t.Fatalf("Failed to encode %+v", priorities)
		}
	} else {
		http.Error(w, "Unknown method", http.StatusNotFound)
	}
}

func (e *Extender) Filter(pod *api.Pod, nodes *api.NodeList) (*api.NodeList, schedulerapi.FailedNodesMap, error) {
	filtered := []api.Node{}
	failedNodesMap := schedulerapi.FailedNodesMap{}
	for _, node := range nodes.Items {
		fits := true
		for _, predicate := range e.predicates {
			fit, err := predicate(pod, &node)
			if err != nil {
				return &api.NodeList{}, schedulerapi.FailedNodesMap{}, err
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
	return &api.NodeList{Items: filtered}, failedNodesMap, nil
}

func (e *Extender) Prioritize(pod *api.Pod, nodes *api.NodeList) (*schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}
	combinedScores := map[string]int{}
	for _, prioritizer := range e.prioritizers {
		weight := prioritizer.weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.function
		prioritizedList, err := priorityFunc(pod, nodes)
		if err != nil {
			return &schedulerapi.HostPriorityList{}, err
		}
		for _, hostEntry := range *prioritizedList {
			combinedScores[hostEntry.Host] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		result = append(result, schedulerapi.HostPriority{Host: host, Score: score})
	}
	return &result, nil
}

func machine_1_2_3_Predicate(pod *api.Pod, node *api.Node) (bool, error) {
	if node.Name == "machine1" || node.Name == "machine2" || node.Name == "machine3" {
		return true, nil
	}
	return false, nil
}

func machine_2_3_5_Predicate(pod *api.Pod, node *api.Node) (bool, error) {
	if node.Name == "machine2" || node.Name == "machine3" || node.Name == "machine5" {
		return true, nil
	}
	return false, nil
}

func machine_2_Prioritizer(pod *api.Pod, nodes *api.NodeList) (*schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}
	for _, node := range nodes.Items {
		score := 1
		if node.Name == "machine2" {
			score = 10
		}
		result = append(result, schedulerapi.HostPriority{node.Name, score})
	}
	return &result, nil
}

func machine_3_Prioritizer(pod *api.Pod, nodes *api.NodeList) (*schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}
	for _, node := range nodes.Items {
		score := 1
		if node.Name == "machine3" {
			score = 10
		}
		result = append(result, schedulerapi.HostPriority{node.Name, score})
	}
	return &result, nil
}

func TestSchedulerExtender(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("scheduler-extender", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})

	extender1 := &Extender{
		name:         "extender1",
		predicates:   []fitPredicate{machine_1_2_3_Predicate},
		prioritizers: []priorityConfig{{machine_2_Prioritizer, 1}},
	}
	es1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		extender1.serveHTTP(t, w, req)
	}))
	defer es1.Close()

	extender2 := &Extender{
		name:         "extender2",
		predicates:   []fitPredicate{machine_2_3_5_Predicate},
		prioritizers: []priorityConfig{{machine_3_Prioritizer, 1}},
	}
	es2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		extender2.serveHTTP(t, w, req)
	}))
	defer es2.Close()

	policy := schedulerapi.Policy{
		ExtenderConfigs: []schedulerapi.ExtenderConfig{
			{
				URLPrefix:      es1.URL,
				FilterVerb:     filter,
				PrioritizeVerb: prioritize,
				Weight:         3,
				EnableHttps:    false,
			},
			{
				URLPrefix:      es2.URL,
				FilterVerb:     filter,
				PrioritizeVerb: prioritize,
				Weight:         4,
				EnableHttps:    false,
			},
		},
	}
	policy.APIVersion = registered.GroupOrDie(api.GroupName).GroupVersion.String()

	schedulerConfigFactory := factory.NewConfigFactory(clientSet, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.CreateFromConfig(policy)
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: api.DefaultSchedulerName})
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: clientSet.Core().Events("")})
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	DoTestPodScheduling(ns, t, clientSet)
}

func DoTestPodScheduling(ns *api.Namespace, t *testing.T, cs clientset.Interface) {
	// NOTE: This test cannot run in parallel, because it is creating and deleting
	// non-namespaced objects (Nodes).
	defer cs.Core().Nodes().DeleteCollection(nil, api.ListOptions{})

	goodCondition := api.NodeCondition{
		Type:              api.NodeReady,
		Status:            api.ConditionTrue,
		Reason:            fmt.Sprintf("schedulable condition"),
		LastHeartbeatTime: unversioned.Time{time.Now()},
	}
	node := &api.Node{
		Spec: api.NodeSpec{Unschedulable: false},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods: *resource.NewQuantity(32, resource.DecimalSI),
			},
			Conditions: []api.NodeCondition{goodCondition},
		},
	}

	for ii := 0; ii < 5; ii++ {
		node.Name = fmt.Sprintf("machine%d", ii+1)
		if _, err := cs.Core().Nodes().Create(node); err != nil {
			t.Fatalf("Failed to create nodes: %v", err)
		}
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "extender-test-pod"},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "container", Image: e2e.GetPauseImageName(cs)}},
		},
	}

	myPod, err := cs.Core().Pods(ns.Name).Create(pod)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	err = wait.Poll(time.Second, wait.ForeverTestTimeout, podScheduled(cs, myPod.Namespace, myPod.Name))
	if err != nil {
		t.Fatalf("Failed to schedule pod: %v", err)
	}

	if myPod, err := cs.Core().Pods(ns.Name).Get(myPod.Name); err != nil {
		t.Fatalf("Failed to get pod: %v", err)
	} else if myPod.Spec.NodeName != "machine3" {
		t.Fatalf("Failed to schedule using extender, expected machine3, got %v", myPod.Spec.NodeName)
	}
	t.Logf("Scheduled pod using extenders")
}
