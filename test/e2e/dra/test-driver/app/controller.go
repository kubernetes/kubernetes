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
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/dynamic-resource-allocation/controller"
	"k8s.io/klog/v2"
)

type Resources struct {
	DriverName         string
	DontSetReservedFor bool
	NodeLocal          bool
	// Nodes is a fixed list of node names on which resources are
	// available. Mutually exclusive with NodeLabels.
	Nodes []string
	// NodeLabels are labels which determine on which nodes resources are
	// available. Mutually exclusive with Nodes.
	NodeLabels     labels.Set
	MaxAllocations int
	Shareable      bool

	// AllocateWrapper, if set, gets called for each Allocate call.
	AllocateWrapper AllocateWrapperType
}

func (r Resources) AllNodes(nodeLister listersv1.NodeLister) []string {
	if len(r.NodeLabels) > 0 {
		// Determine nodes with resources dynamically.
		nodes, _ := nodeLister.List(labels.SelectorFromValidatedSet(r.NodeLabels))
		nodeNames := make([]string, 0, len(nodes))
		for _, node := range nodes {
			nodeNames = append(nodeNames, node.Name)
		}
		return nodeNames
	}
	return r.Nodes
}

func (r Resources) NewAllocation(node string, data []byte) *resourcev1alpha2.AllocationResult {
	allocation := &resourcev1alpha2.AllocationResult{
		Shareable: r.Shareable,
	}
	allocation.ResourceHandles = []resourcev1alpha2.ResourceHandle{
		{
			DriverName: r.DriverName,
			Data:       string(data),
		},
	}
	if node == "" && len(r.NodeLabels) > 0 {
		// Available on all nodes matching the labels.
		var requirements []v1.NodeSelectorRequirement
		for key, value := range r.NodeLabels {
			requirements = append(requirements, v1.NodeSelectorRequirement{
				Key:      key,
				Operator: v1.NodeSelectorOpIn,
				Values:   []string{value},
			})
		}
		allocation.AvailableOnNodes = &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: requirements,
				},
			},
		}
	} else {
		var nodes []string
		if node != "" {
			// Local to one node.
			nodes = append(nodes, node)
		} else {
			// Available on the fixed set of nodes.
			nodes = r.Nodes
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
	}

	return allocation
}

type AllocateWrapperType func(ctx context.Context, claimAllocations []*controller.ClaimAllocation,
	selectedNode string,
	handler func(ctx context.Context,
		claimAllocations []*controller.ClaimAllocation,
		selectedNode string),
)

type ExampleController struct {
	clientset  kubernetes.Interface
	nodeLister listersv1.NodeLister
	resources  Resources

	mutex sync.Mutex
	// allocated maps claim.UID to the node (if network-attached) or empty (if not).
	allocated map[types.UID]string
	// claimsPerNode counts how many claims are currently allocated for a node (non-empty key)
	// or allocated for the entire cluster (empty key). Must be kept in sync with allocated.
	claimsPerNode map[string]int

	numAllocations, numDeallocations int64
}

func NewController(clientset kubernetes.Interface, resources Resources) *ExampleController {
	c := &ExampleController{
		clientset: clientset,
		resources: resources,

		allocated:     make(map[types.UID]string),
		claimsPerNode: make(map[string]int),
	}
	return c
}

func (c *ExampleController) Run(ctx context.Context, workers int) {
	informerFactory := informers.NewSharedInformerFactory(c.clientset, 0 /* resync period */)
	ctrl := controller.New(ctx, c.resources.DriverName, c, c.clientset, informerFactory)
	c.nodeLister = informerFactory.Core().V1().Nodes().Lister()
	ctrl.SetReservedFor(!c.resources.DontSetReservedFor)
	informerFactory.Start(ctx.Done())
	ctrl.Run(workers)
	// If we get here, the context was canceled and we can wait for informer factory goroutines.
	informerFactory.Shutdown()
}

type parameters struct {
	EnvVars  map[string]string
	NodeName string
}

var _ controller.Driver = &ExampleController{}

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

func (c *ExampleController) GetClassParameters(ctx context.Context, class *resourcev1alpha2.ResourceClass) (interface{}, error) {
	if class.ParametersRef != nil {
		if class.ParametersRef.APIGroup != "" ||
			class.ParametersRef.Kind != "ConfigMap" {
			return nil, fmt.Errorf("class parameters are only supported in APIVersion v1, Kind ConfigMap, got: %v", class.ParametersRef)
		}
		return c.readParametersFromConfigMap(ctx, class.ParametersRef.Namespace, class.ParametersRef.Name)
	}
	return nil, nil
}

func (c *ExampleController) GetClaimParameters(ctx context.Context, claim *resourcev1alpha2.ResourceClaim, class *resourcev1alpha2.ResourceClass, classParameters interface{}) (interface{}, error) {
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
		return nil, fmt.Errorf("get config map: %w", err)
	}
	return configMap.Data, nil
}

func (c *ExampleController) Allocate(ctx context.Context, claimAllocations []*controller.ClaimAllocation, selectedNode string) {

	if c.resources.AllocateWrapper != nil {
		c.resources.AllocateWrapper(ctx, claimAllocations, selectedNode, c.allocateOneByOne)
	} else {
		c.allocateOneByOne(ctx, claimAllocations, selectedNode)
	}

	return
}

func (c *ExampleController) allocateOneByOne(ctx context.Context, claimAllocations []*controller.ClaimAllocation, selectedNode string) {
	for _, ca := range claimAllocations {
		allocationResult, err := c.allocateOne(ctx, ca.Claim, ca.ClaimParameters, ca.Class, ca.ClassParameters, selectedNode)
		if err != nil {
			ca.Error = fmt.Errorf("failed allocating claim %v", ca.Claim.UID)
			continue
		}
		ca.Allocation = allocationResult
	}
}

// allocate simply copies parameters as JSON map into a ResourceHandle.
func (c *ExampleController) allocateOne(ctx context.Context, claim *resourcev1alpha2.ResourceClaim, claimParameters interface{}, class *resourcev1alpha2.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha2.AllocationResult, err error) {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "Allocate"), "claim", klog.KObj(claim), "uid", claim.UID)
	defer func() {
		logger.V(3).Info("done", "result", result, "err", err)
	}()

	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Already allocated? Then we don't need to count it again.
	node, alreadyAllocated := c.allocated[claim.UID]
	if alreadyAllocated {
		// Idempotent result - kind of. We don't check whether
		// the parameters changed in the meantime. A real
		// driver would have to do that.
		logger.V(3).V(3).Info("already allocated")
	} else {
		logger.V(3).Info("starting", "selectedNode", selectedNode)
		nodes := c.resources.AllNodes(c.nodeLister)
		if c.resources.NodeLocal {
			node = selectedNode
			if node == "" {
				// If none has been selected because we do immediate allocation,
				// then we need to pick one ourselves.
				var viableNodes []string
				for _, n := range nodes {
					if c.resources.MaxAllocations == 0 ||
						c.claimsPerNode[n] < c.resources.MaxAllocations {
						viableNodes = append(viableNodes, n)
					}
				}
				if len(viableNodes) == 0 {
					return nil, errors.New("resources exhausted on all nodes")
				}
				// Pick randomly. We could also prefer the one with the least
				// number of allocations (even spreading) or the most (packing).
				node = viableNodes[rand.Intn(len(viableNodes))]
				logger.V(3).Info("picked a node ourselves", "selectedNode", selectedNode)
			} else if !contains(nodes, node) ||
				c.resources.MaxAllocations > 0 &&
					c.claimsPerNode[node] >= c.resources.MaxAllocations {
				return nil, fmt.Errorf("resources exhausted on node %q", node)
			}
		} else {
			if c.resources.MaxAllocations > 0 &&
				len(c.allocated) >= c.resources.MaxAllocations {
				return nil, errors.New("resources exhausted in the cluster")
			}
		}
	}

	p := parameters{
		EnvVars:  make(map[string]string),
		NodeName: node,
	}
	toEnvVars("user", claimParameters, p.EnvVars)
	toEnvVars("admin", classParameters, p.EnvVars)
	data, err := json.Marshal(p)
	if err != nil {
		return nil, fmt.Errorf("encode parameters: %w", err)
	}
	allocation := c.resources.NewAllocation(node, data)
	if !alreadyAllocated {
		c.numAllocations++
		c.allocated[claim.UID] = node
		c.claimsPerNode[node]++
	}
	return allocation, nil
}

func (c *ExampleController) Deallocate(ctx context.Context, claim *resourcev1alpha2.ResourceClaim) error {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "Deallocate"), "claim", klog.KObj(claim), "uid", claim.UID)
	c.mutex.Lock()
	defer c.mutex.Unlock()

	node, ok := c.allocated[claim.UID]
	if !ok {
		logger.V(3).Info("already deallocated")
		return nil
	}

	logger.V(3).Info("done")
	c.numDeallocations++
	delete(c.allocated, claim.UID)
	c.claimsPerNode[node]--
	return nil
}

func (c *ExampleController) UnsuitableNodes(ctx context.Context, pod *v1.Pod, claims []*controller.ClaimAllocation, potentialNodes []string) (finalErr error) {
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.FromContext(ctx), "UnsuitableNodes"), "pod", klog.KObj(pod))
	c.mutex.Lock()
	defer c.mutex.Unlock()

	logger.V(3).Info("starting", "claims", claims, "potentialNodes", potentialNodes)
	defer func() {
		// UnsuitableNodes is the same for all claims.
		logger.V(3).Info("done", "unsuitableNodes", claims[0].UnsuitableNodes, "err", finalErr)
	}()

	if c.resources.MaxAllocations == 0 {
		// All nodes are suitable.
		return nil
	}
	nodes := c.resources.AllNodes(c.nodeLister)
	if c.resources.NodeLocal {
		for _, claim := range claims {
			claim.UnsuitableNodes = nil
			for _, node := range potentialNodes {
				// If we have more than one claim, then a
				// single pod wants to use all of them.  That
				// can only work if a node has capacity left
				// for all of them. Also, nodes that the driver
				// doesn't run on cannot be used.
				if !contains(nodes, node) ||
					c.claimsPerNode[node]+len(claims) > c.resources.MaxAllocations {
					claim.UnsuitableNodes = append(claim.UnsuitableNodes, node)
				}
			}
		}
		return nil
	}

	allocations := c.claimsPerNode[""]
	for _, claim := range claims {
		claim.UnsuitableNodes = nil
		for _, node := range potentialNodes {
			if !contains(nodes, node) ||
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
