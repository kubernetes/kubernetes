/*
Copyright 2022 The Kubernetes Authors.

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

// Package app does all of the work necessary to configure and run a
// Kubernetes app process.
package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/controller"
	"k8s.io/klog/v2"
)

type Resources struct {
	NodeLocal      bool
	Nodes          []string
	MaxAllocations int
	Shareable      bool

	// AllocateWrapper, if set, gets called for each Allocate call.
	AllocateWrapper AllocateWrapperType
}

type AllocateWrapperType func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{},
	class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string,
	handler func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{},
		class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha1.AllocationResult, err error),
) (result *resourcev1alpha1.AllocationResult, err error)

type ExampleController struct {
	clientset  kubernetes.Interface
	resources  Resources
	driverName string

	// mutex must be locked at the gRPC call level.
	mutex sync.Mutex
	// allocated maps claim.UID to the node (if network-attached) or empty (if not).
	allocated map[types.UID]string

	numAllocations, numDeallocations int64
}

func NewController(clientset kubernetes.Interface, driverName string, resources Resources) *ExampleController {
	c := &ExampleController{
		clientset:  clientset,
		resources:  resources,
		driverName: driverName,

		allocated: make(map[types.UID]string),
	}
	return c
}

func (c *ExampleController) Run(ctx context.Context, workers int) *ExampleController {
	informerFactory := informers.NewSharedInformerFactory(c.clientset, 0 /* resync period */)
	ctrl := controller.New(ctx, c.driverName, c, c.clientset, informerFactory)
	informerFactory.Start(ctx.Done())
	ctrl.Run(workers)

	return c
}

type parameters struct {
	EnvVars  map[string]string
	NodeName string
}

var _ controller.Driver = &ExampleController{}

func (c *ExampleController) countAllocations(node string) int {
	total := 0
	for _, n := range c.allocated {
		if n == node {
			total++
		}
	}
	return total
}

// GetNumAllocations returns the number of times that a claim was allocated.
// Idempotent calls to Allocate that do not need to allocate the claim again do
// not contribute to that counter.
func (c *ExampleController) GetNumAllocations() int64 {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.numAllocations
}

// GetNumDeallocations returns the number of times that a claim was allocated.
// Idempotent calls to Allocate that do not need to allocate the claim again do
// not contribute to that counter.
func (c *ExampleController) GetNumDeallocations() int64 {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.numDeallocations
}

func (c *ExampleController) GetClassParameters(ctx context.Context, class *resourcev1alpha1.ResourceClass) (interface{}, error) {
	if class.ParametersRef != nil {
		if class.ParametersRef.APIGroup != "" ||
			class.ParametersRef.Kind != "ConfigMap" {
			return nil, fmt.Errorf("class parameters are only supported in APIVersion v1, Kind ConfigMap, got: %v", class.ParametersRef)
		}
		return c.readParametersFromConfigMap(ctx, class.ParametersRef.Namespace, class.ParametersRef.Name)
	}
	return nil, nil
}

func (c *ExampleController) GetClaimParameters(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, class *resourcev1alpha1.ResourceClass, classParameters interface{}) (interface{}, error) {
	if claim.Spec.ParametersRef != nil {
		if claim.Spec.ParametersRef.APIGroup != "" ||
			claim.Spec.ParametersRef.Kind != "ConfigMap" {
			return nil, fmt.Errorf("claim parameters are only supported in APIVersion v1, Kind ConfigMap, got: %v", claim.Spec.ParametersRef)
		}
		return c.readParametersFromConfigMap(ctx, claim.Namespace, claim.Spec.ParametersRef.Name)
	}
	return nil, nil
}

func (c *ExampleController) readParametersFromConfigMap(ctx context.Context, namespace, name string) (map[string]string, error) {
	configMap, err := c.clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get config map: %v", err)
	}
	return configMap.Data, nil
}

func (c *ExampleController) Allocate(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{}, class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha1.AllocationResult, err error) {
	if c.resources.AllocateWrapper != nil {
		return c.resources.AllocateWrapper(ctx, claim, claimParameters, class, classParameters, selectedNode, c.allocate)
	}
	return c.allocate(ctx, claim, claimParameters, class, classParameters, selectedNode)
}

// allocate simply copies parameters as JSON map into ResourceHandle.
func (c *ExampleController) allocate(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{}, class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha1.AllocationResult, err error) {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "Allocate"), "claim", klog.KObj(claim), "uid", claim.UID)
	defer func() {
		logger.Info("done", "result", prettyPrint(result), "err", err)
	}()

	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Already allocated? Then we don't need to count it again.
	node, alreadyAllocated := c.allocated[claim.UID]
	if alreadyAllocated {
		// Idempotent result - kind of. We don't check whether
		// the parameters changed in the meantime. A real
		// driver would have to do that.
		logger.Info("already allocated")
	} else {
		logger.Info("starting", "selectedNode", selectedNode)
		if c.resources.NodeLocal {
			node = selectedNode
			if node == "" {
				// If none has been selected because we do immediate allocation,
				// then we need to pick one ourselves.
				var viableNodes []string
				for _, n := range c.resources.Nodes {
					if c.resources.MaxAllocations == 0 ||
						c.countAllocations(n) < c.resources.MaxAllocations {
						viableNodes = append(viableNodes, n)
					}
				}
				if len(viableNodes) == 0 {
					return nil, errors.New("resources exhausted on all nodes")
				}
				// Pick randomly. We could also prefer the one with the least
				// number of allocations (even spreading) or the most (packing).
				node = viableNodes[rand.Intn(len(viableNodes))]
				logger.Info("picked a node ourselves", "selectedNode", selectedNode)
			} else if c.resources.MaxAllocations > 0 &&
				c.countAllocations(node) >= c.resources.MaxAllocations {
				return nil, fmt.Errorf("resources exhausted on node %q", node)
			}
		} else {
			if c.resources.MaxAllocations > 0 &&
				len(c.allocated) >= c.resources.MaxAllocations {
				return nil, errors.New("resources exhausted in the cluster")
			}
		}
	}

	allocation := &resourcev1alpha1.AllocationResult{
		Shareable: c.resources.Shareable,
	}
	p := parameters{
		EnvVars:  make(map[string]string),
		NodeName: node,
	}
	toEnvVars("user", claimParameters, p.EnvVars)
	toEnvVars("admin", classParameters, p.EnvVars)
	data, err := json.Marshal(p)
	if err != nil {
		return nil, fmt.Errorf("encode parameters: %v", err)
	}
	allocation.ResourceHandle = string(data)
	var nodes []string
	if node != "" {
		nodes = append(nodes, node)
	} else {
		nodes = c.resources.Nodes
	}
	if len(nodes) > 0 {
		allocation.AvailableOnNodes = &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "kubernetes.io/hostname",
							Operator: v1.NodeSelectorOpIn,
							Values:   nodes,
						},
					},
				},
			},
		}
	}
	if !alreadyAllocated {
		c.numAllocations++
		c.allocated[claim.UID] = node
	}
	return allocation, nil
}

func (c *ExampleController) Deallocate(ctx context.Context, claim *resourcev1alpha1.ResourceClaim) error {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "Deallocate"), "claim", klog.KObj(claim), "uid", claim.UID)
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if _, ok := c.allocated[claim.UID]; !ok {
		logger.Info("already deallocated")
		return nil
	}

	logger.Info("done")
	c.numDeallocations++
	delete(c.allocated, claim.UID)
	return nil
}

func (c *ExampleController) UnsuitableNodes(ctx context.Context, pod *v1.Pod, claims []*controller.ClaimAllocation, potentialNodes []string) (finalErr error) {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "UnsuitableNodes"), "pod", klog.KObj(pod))
	logger.Info("starting", "claim", prettyPrintSlice(claims), "potentialNodes", potentialNodes)
	defer func() {
		// UnsuitableNodes is the same for all claims.
		logger.Info("done", "unsuitableNodes", claims[0].UnsuitableNodes, "err", finalErr)
	}()
	if c.resources.MaxAllocations == 0 {
		// All nodes are suitable.
		return nil
	}
	if c.resources.NodeLocal {
		allocationsPerNode := make(map[string]int)
		for _, node := range c.resources.Nodes {
			allocationsPerNode[node] = c.countAllocations(node)
		}
		for _, claim := range claims {
			claim.UnsuitableNodes = nil
			for _, node := range potentialNodes {
				// If we have more than one claim, then a
				// single pod wants to use all of them.  That
				// can only work if a node has capacity left
				// for all of them. Also, nodes that the driver
				// doesn't run on cannot be used.
				if contains(c.resources.Nodes, node) &&
					allocationsPerNode[node]+len(claims) > c.resources.MaxAllocations {
					claim.UnsuitableNodes = append(claim.UnsuitableNodes, node)
				}
			}
		}
		return nil
	}

	allocations := c.countAllocations("")
	for _, claim := range claims {
		claim.UnsuitableNodes = nil
		for _, node := range potentialNodes {
			if contains(c.resources.Nodes, node) &&
				allocations+len(claims) > c.resources.MaxAllocations {
				claim.UnsuitableNodes = append(claim.UnsuitableNodes, node)
			}
		}
	}

	return nil
}

func toEnvVars(what string, from interface{}, to map[string]string) {
	if from == nil {
		return
	}

	env := from.(map[string]string)
	for key, value := range env {
		to[what+"_"+strings.ToLower(key)] = value
	}
}

func contains[T comparable](list []T, value T) bool {
	for _, v := range list {
		if v == value {
			return true
		}
	}

	return false
}

func prettyPrint[T any](obj *T) interface{} {
	if obj == nil {
		return "<nil>"
	}
	return *obj
}

// prettyPrintSlice prints the values the slice points to, not the pointers.
func prettyPrintSlice[T any](slice []*T) interface{} {
	var values []interface{}
	for _, v := range slice {
		if v == nil {
			values = append(values, "<nil>")
		} else {
			values = append(values, *v)
		}
	}
	return values
}
