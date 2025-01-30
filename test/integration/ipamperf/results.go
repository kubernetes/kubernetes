/*
Copyright 2018 The Kubernetes Authors.

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

package ipamperf

import (
	"bytes"
	"fmt"
	"sort"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
)

// Config represents the test configuration that is being run
type Config struct {
	CreateQPS     int                     // rate at which nodes are created
	KubeQPS       int                     // rate for communication with kubernetes API
	CloudQPS      int                     // rate for communication with cloud endpoint
	NumNodes      int                     // number of nodes to created and monitored
	AllocatorType ipam.CIDRAllocatorType  // type of allocator to run
	Cloud         cloudprovider.Interface // cloud provider
}

type nodeTime struct {
	added     time.Time // observed time for when node was added
	allocated time.Time // observed time for when node was assigned podCIDR
	podCIDR   string    // the allocated podCIDR range
}

// Observer represents the handle to test observer that watches for node changes
// and tracks behavior
type Observer struct {
	numAdded     int                  // number of nodes observed added
	numAllocated int                  // number of nodes observed allocated podCIDR
	timing       map[string]*nodeTime // per node timing
	numNodes     int                  // the number of nodes to expect
	stopChan     chan struct{}        // for the shared informer
	wg           sync.WaitGroup
	clientSet    *clientset.Clientset
}

// JSONDuration is an alias of time.Duration to support custom Marshal code
type JSONDuration time.Duration

// NodeDuration represents the CIDR allocation time for each node
type NodeDuration struct {
	Name     string       // node name
	PodCIDR  string       // the podCIDR that was assigned to the node
	Duration JSONDuration // how long it took to assign podCIDR
}

// Results represents the observed test results.
type Results struct {
	Name           string         // name for the test
	Config         *Config        // handle to the test config
	Succeeded      bool           // whether all nodes were assigned podCIDR
	MaxAllocTime   JSONDuration   // the maximum time take for assignment per node
	TotalAllocTime JSONDuration   // duration between first addition and last assignment
	NodeAllocTime  []NodeDuration // assignment time by node name
}

// NewObserver creates a new observer given a handle to the Clientset
func NewObserver(clientSet *clientset.Clientset, numNodes int) *Observer {
	o := &Observer{
		timing:    map[string]*nodeTime{},
		numNodes:  numNodes,
		clientSet: clientSet,
		stopChan:  make(chan struct{}),
	}
	return o
}

// StartObserving starts an asynchronous loop to monitor for node changes.
// Call Results() to get the test results after starting observer.
func (o *Observer) StartObserving() error {
	o.monitor()
	klog.Infof("Test observer started")
	return nil
}

// Results returns the test results. It waits for the observer to finish
// and returns the computed results of the observations.
func (o *Observer) Results(name string, config *Config) *Results {
	var (
		firstAdd       time.Time // earliest time any node was added (first node add)
		lastAssignment time.Time // latest time any node was assigned CIDR (last node assignment)
	)
	o.wg.Wait()
	close(o.stopChan) // shutdown the shared informer

	results := &Results{
		Name:          name,
		Config:        config,
		Succeeded:     o.numAdded == o.numNodes && o.numAllocated == o.numNodes,
		MaxAllocTime:  0,
		NodeAllocTime: []NodeDuration{},
	}
	for name, nTime := range o.timing {
		addFound := !nTime.added.IsZero()
		if addFound && (firstAdd.IsZero() || nTime.added.Before(firstAdd)) {
			firstAdd = nTime.added
		}
		cidrFound := !nTime.allocated.IsZero()
		if cidrFound && nTime.allocated.After(lastAssignment) {
			lastAssignment = nTime.allocated
		}
		if addFound && cidrFound {
			allocTime := nTime.allocated.Sub(nTime.added)
			if allocTime > time.Duration(results.MaxAllocTime) {
				results.MaxAllocTime = JSONDuration(allocTime)
			}
			results.NodeAllocTime = append(results.NodeAllocTime, NodeDuration{
				Name: name, PodCIDR: nTime.podCIDR, Duration: JSONDuration(allocTime),
			})
		}
	}
	results.TotalAllocTime = JSONDuration(lastAssignment.Sub(firstAdd))
	sort.Slice(results.NodeAllocTime, func(i, j int) bool {
		return results.NodeAllocTime[i].Duration > results.NodeAllocTime[j].Duration
	})
	return results
}

func (o *Observer) monitor() {
	o.wg.Add(1)

	sharedInformer := informers.NewSharedInformerFactory(o.clientSet, 1*time.Second)
	nodeInformer := sharedInformer.Core().V1().Nodes().Informer()

	nodeInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(func(node *v1.Node) (err error) {
			name := node.GetName()
			if node.Spec.PodCIDR != "" {
				// ignore nodes that have PodCIDR (might be hold over from previous runs that did not get cleaned up)
				return
			}
			nTime := &nodeTime{}
			o.timing[name] = nTime
			nTime.added = time.Now()
			o.numAdded = o.numAdded + 1
			return
		}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(oldNode, newNode *v1.Node) (err error) {
			name := newNode.GetName()
			nTime, found := o.timing[name]
			if !found {
				return // consistency check - ignore nodes we have not seen the add event for
			}
			// check if CIDR assigned and ignore redundant updates
			if newNode.Spec.PodCIDR != "" && nTime.podCIDR == "" {
				nTime.allocated = time.Now()
				nTime.podCIDR = newNode.Spec.PodCIDR
				o.numAllocated++
				if o.numAllocated%10 == 0 {
					klog.Infof("progress: %d/%d - %.2d%%", o.numAllocated, o.numNodes, (o.numAllocated * 100.0 / o.numNodes))
				}
				// do following check only if numAllocated is modified, as otherwise, redundant updates
				// can cause wg.Done() to be called multiple times, causing a panic
				if o.numAdded == o.numNodes && o.numAllocated == o.numNodes {
					klog.Info("All nodes assigned podCIDR")
					o.wg.Done()
				}
			}
			return
		}),
	})
	sharedInformer.Start(o.stopChan)
}

// String implements the Stringer interface and returns a multi-line representation
// of the test results.
func (results *Results) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "\n  TestName: %s", results.Name)
	fmt.Fprintf(&b, "\n  NumNodes: %d, CreateQPS: %d, KubeQPS: %d, CloudQPS: %d, Allocator: %v",
		results.Config.NumNodes, results.Config.CreateQPS, results.Config.KubeQPS,
		results.Config.CloudQPS, results.Config.AllocatorType)
	fmt.Fprintf(&b, "\n  Succeeded: %v, TotalAllocTime: %v, MaxAllocTime: %v",
		results.Succeeded, time.Duration(results.TotalAllocTime), time.Duration(results.MaxAllocTime))
	fmt.Fprintf(&b, "\n  %5s %-20s %-20s %s", "Num", "Node", "PodCIDR", "Duration (s)")
	for i, d := range results.NodeAllocTime {
		fmt.Fprintf(&b, "\n  %5d %-20s %-20s %10.3f", i+1, d.Name, d.PodCIDR, time.Duration(d.Duration).Seconds())
	}
	return b.String()
}

// MarshalJSON implements the json.Marshaler interface
func (jDuration *JSONDuration) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf("\"%s\"", time.Duration(*jDuration).String())), nil
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (jDuration *JSONDuration) UnmarshalJSON(b []byte) (err error) {
	var d time.Duration
	if d, err = time.ParseDuration(string(b[1 : len(b)-1])); err == nil {
		*jDuration = JSONDuration(d)
	}
	return
}
