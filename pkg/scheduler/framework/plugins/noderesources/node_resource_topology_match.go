/*
Copyright 2020 The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"fmt"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog/v2"
	bm "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	topologyv1alpha1 "k8s.io/noderesourcetopology-api/pkg/apis/topology/v1alpha1"
	topoclientset "k8s.io/noderesourcetopology-api/pkg/generated/clientset/versioned"
	topoinformerexternal "k8s.io/noderesourcetopology-api/pkg/generated/informers/externalversions"
	topologyinformers "k8s.io/noderesourcetopology-api/pkg/generated/informers/externalversions"
	topoinformerv1alpha1 "k8s.io/noderesourcetopology-api/pkg/generated/informers/externalversions/topology/v1alpha1"
)

const (
	// NodeResourceTopologyMatchName is the name of the plugin used in the plugin registry and configurations.
	NodeResourceTopologyMatchName = "NodeResourceTopologyMatch"
)

var _ framework.FilterPlugin = &NodeResourceTopologyMatch{}

type nodeTopologyMap map[string]topologyv1alpha1.NodeResourceTopology

type PolicyHandler interface {
	PolicyFilter(pod *v1.Pod, zoneMap topologyv1alpha1.ZoneMap) *framework.Status
}

type PolicyHandlerMap map[v1.TopologyManagerPolicy]PolicyHandler

// NodeResourceTopologyMatch plugin which run simplified version of TopologyManager's admit handler
type NodeResourceTopologyMatch struct {
	handle framework.Handle

	nodeTopologyInformer    topoinformerv1alpha1.NodeResourceTopologyInformer
	topologyInformerFactory topoinformerexternal.SharedInformerFactory

	nodeTopologies         nodeTopologyMap
	nodeTopologyGuard      sync.RWMutex
	topologyPolicyHandlers PolicyHandlerMap
}

type NUMANode struct {
	NUMAID int
	Resources v1.ResourceList
}

type NUMANodeList []NUMANode

// Name returns name of the plugin. It is used in logs, etc.
func (tm *NodeResourceTopologyMatch) Name() string {
	return NodeResourceTopologyMatchName
}

func filter(containers []v1.Container, nodes NUMANodeList) *framework.Status {
	for _, container := range containers  {
		bitmask := bm.NewEmptyBitMask()
		bitmask.Fill()
		for resource, quantity := range container.Resources.Requests {
			resourceBitmask := bm.NewEmptyBitMask()
			for _, numaNode := range nodes {
				numaQuantity, ok := numaNode.Resources[resource]
				if !ok {
					continue
				}
				// Check for the following:
				// 1. set numa node as possible node if resource is memory or Hugepages (until memory manager will not be merged and
				// memory will not be provided in CRD
				// 2. otherwise check amount of resources
				if resource == v1.ResourceMemory ||
					strings.HasPrefix(string(resource), string(v1.ResourceHugePagesPrefix)) ||
					numaQuantity.Cmp(quantity) >= 0 {
					resourceBitmask.Add(numaNode.NUMAID)
				}
			}
			bitmask.And(resourceBitmask)
		}
		if bitmask.IsEmpty() {
			// definitly we can't align container, so we can't align a pod
			return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Can't align container: %s", container.Name))
		}
	}
	return nil
}

// checkTopologyPolicies return true if we're working with such policy
func getTopologyPolicies(nodeTopologies nodeTopologyMap, nodeName string) []v1.TopologyManagerPolicy {
	if nodeTopology, ok := nodeTopologies[nodeName]; ok {
		policies := make([]v1.TopologyManagerPolicy, 0)
		for _, policy := range nodeTopology.TopologyPolicies {
			policies = append(policies, v1.TopologyManagerPolicy(policy))
		}
		return policies
	}
	return nil
}
func extractResources(zone topologyv1alpha1.Zone) v1.ResourceList {
	res := make(v1.ResourceList)
	for resName, resInfo := range zone.Resources {
		quantity, err := resource.ParseQuantity(resInfo.Allocatable.String())
		if err != nil {
			klog.Errorf("Failed to parse %s", resInfo.Allocatable.String())
			continue
		}
		res[resName] = quantity
	}
	return res
}

func (tm NodeResourceTopologyMatch) PolicyFilter(pod *v1.Pod, zoneMap topologyv1alpha1.ZoneMap) *framework.Status {
	containers := []v1.Container{}
	containers = append(pod.Spec.InitContainers, pod.Spec.Containers...)

	tm.nodeTopologyGuard.RLock()
	defer tm.nodeTopologyGuard.RUnlock()
	// prepare NUMANodes list from zoneMap

	nodes := make(NUMANodeList, 0)
	for zoneName, zone := range zoneMap {
		if zone.Type == "Node" {
			var numaID int
			fmt.Sscanf(zoneName, "node-%d", &numaID)
			resources := extractResources(zone)
			nodes = append(nodes, NUMANode{NUMAID: numaID, Resources: resources})
		}
	}
	return filter(containers, nodes)
}

// Filter Now only single-numa-node supported
func (tm *NodeResourceTopologyMatch) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if nodeInfo.Node() == nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("Node is nil %s", nodeInfo.Node().Name))
	}
	nodeName := nodeInfo.Node().Name

	topologyPolicies := getTopologyPolicies(tm.nodeTopologies, nodeName)
	for _, policyName := range topologyPolicies {
		if handler, ok := tm.topologyPolicyHandlers[policyName]; ok {
			if status := handler.PolicyFilter(pod, tm.nodeTopologies[nodeName].Zones); status != nil {
				return status
			}
		} else {
			klog.V(5).Infof("Handler for policy %s not found", policyName)
		}
	}
	return nil
}

func (tm *NodeResourceTopologyMatch) onTopologyCRDFromDelete(obj interface{}) {
	var nodeTopology *topologyv1alpha1.NodeResourceTopology
	switch t := obj.(type) {
	case *topologyv1alpha1.NodeResourceTopology:
		nodeTopology = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		nodeTopology, ok = t.Obj.(*topologyv1alpha1.NodeResourceTopology)
		if !ok {
			klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t.Obj)
			return
		}
	default:
		klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t)
		return
	}

	klog.V(5).Infof("delete event for scheduled NodeResourceTopology %s/%s ",
		nodeTopology.Namespace, nodeTopology.Name)

	tm.nodeTopologyGuard.Lock()
	defer tm.nodeTopologyGuard.Unlock()
	if _, ok := tm.nodeTopologies[nodeTopology.Name]; ok {
		delete(tm.nodeTopologies, nodeTopology.Name)
	}
}

func (tm *NodeResourceTopologyMatch) onTopologyCRDUpdate(oldObj interface{}, newObj interface{}) {
	var nodeTopology *topologyv1alpha1.NodeResourceTopology
	switch t := newObj.(type) {
	case *topologyv1alpha1.NodeResourceTopology:
		nodeTopology = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		nodeTopology, ok = t.Obj.(*topologyv1alpha1.NodeResourceTopology)
		if !ok {
			klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t.Obj)
			return
		}
	default:
		klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t)
		return
	}
	klog.V(5).Infof("update event for scheduled NodeResourceTopology %s/%s ",
		nodeTopology.Namespace, nodeTopology.Name)

	tm.nodeTopologyGuard.Lock()
	defer tm.nodeTopologyGuard.Unlock()
	tm.nodeTopologies[nodeTopology.Name] = *nodeTopology
}

func (tm *NodeResourceTopologyMatch) onTopologyCRDAdd(obj interface{}) {
	var nodeTopology *topologyv1alpha1.NodeResourceTopology
	switch t := obj.(type) {
	case *topologyv1alpha1.NodeResourceTopology:
		nodeTopology = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		nodeTopology, ok = t.Obj.(*topologyv1alpha1.NodeResourceTopology)
		if !ok {
			klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t.Obj)
			return
		}
	default:
		klog.Errorf("cannot convert to *v1alpha1.NodeResourceTopology: %v", t)
		return
	}
	klog.V(5).Infof("add event for scheduled NodeResourceTopology %s/%s ",
		nodeTopology.Namespace, nodeTopology.Name)

	tm.nodeTopologyGuard.Lock()
	defer tm.nodeTopologyGuard.Unlock()
	tm.nodeTopologies[nodeTopology.Name] = *nodeTopology
}

// NewNodeResourceTopologyMatch initializes a new plugin and returns it.
func NewNodeResourceTopologyMatch(args runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	klog.V(5).Infof("creating new NodeResourceTopologyMatch plugin")
        tcfg, ok := args.(*config.NodeResourceTopologyMatchArgs)
        if !ok {
                return nil, fmt.Errorf("want args to be of type NodeResourceTopologyMatchArgs, got %T", args)
        }

	topologyMatch := &NodeResourceTopologyMatch{}

	kubeConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: tcfg.KubeConfig},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: tcfg.MasterOverride}}).ClientConfig()
	if err != nil {
		klog.Errorf("Can't create kubeconfig based on: %s, %s, %v", tcfg.KubeConfig, tcfg.MasterOverride, err)
		return nil, err
	}

	topoClient, err := topoclientset.NewForConfig(kubeConfig)
	if err != nil {
		klog.Errorf("Can't create clientset for NodeTopologyResource: %s, %s", kubeConfig, err)
		return nil, err
	}

	topologyMatch.topologyInformerFactory = topologyinformers.NewSharedInformerFactory(topoClient, 0)
	topologyMatch.nodeTopologyInformer = topologyMatch.topologyInformerFactory.Topology().V1alpha1().NodeResourceTopologies()

	topologyMatch.nodeTopologyInformer.Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    topologyMatch.onTopologyCRDAdd,
			UpdateFunc: topologyMatch.onTopologyCRDUpdate,
			DeleteFunc: topologyMatch.onTopologyCRDFromDelete,
		},
	)

	go topologyMatch.nodeTopologyInformer.Informer().Run(context.Background().Done())
	topologyMatch.topologyInformerFactory.Start(context.Background().Done())

	klog.V(5).Infof("start nodeTopologyInformer")

	topologyMatch.handle = handle

	topologyMatch.topologyPolicyHandlers = make(PolicyHandlerMap)
	topologyMatch.topologyPolicyHandlers[v1.SingleNUMANodeTopologyManagerPolicy] = topologyMatch
	topologyMatch.nodeTopologies = nodeTopologyMap{}

	return topologyMatch, nil
}
