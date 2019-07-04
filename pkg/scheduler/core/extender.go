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

package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

const (
	// DefaultExtenderTimeout defines the default extender timeout in second.
	DefaultExtenderTimeout = 5 * time.Second
)

// HTTPExtender implements the algorithm.SchedulerExtender interface.
type HTTPExtender struct {
	extenderURL      string
	preemptVerb      string
	filterVerb       string
	prioritizeVerb   string
	bindVerb         string
	weight           int
	client           *http.Client
	nodeCacheCapable bool
	managedResources sets.String
	ignorable        bool
}

func makeTransport(config *schedulerapi.ExtenderConfig) (http.RoundTripper, error) {
	var cfg restclient.Config
	if config.TLSConfig != nil {
		cfg.TLSClientConfig.Insecure = config.TLSConfig.Insecure
		cfg.TLSClientConfig.ServerName = config.TLSConfig.ServerName
		cfg.TLSClientConfig.CertFile = config.TLSConfig.CertFile
		cfg.TLSClientConfig.KeyFile = config.TLSConfig.KeyFile
		cfg.TLSClientConfig.CAFile = config.TLSConfig.CAFile
		cfg.TLSClientConfig.CertData = config.TLSConfig.CertData
		cfg.TLSClientConfig.KeyData = config.TLSConfig.KeyData
		cfg.TLSClientConfig.CAData = config.TLSConfig.CAData
	}
	if config.EnableHTTPS {
		hasCA := len(cfg.CAFile) > 0 || len(cfg.CAData) > 0
		if !hasCA {
			cfg.Insecure = true
		}
	}
	tlsConfig, err := restclient.TLSConfigFor(&cfg)
	if err != nil {
		return nil, err
	}
	if tlsConfig != nil {
		return utilnet.SetTransportDefaults(&http.Transport{
			TLSClientConfig: tlsConfig,
		}), nil
	}
	return utilnet.SetTransportDefaults(&http.Transport{}), nil
}

// NewHTTPExtender creates an HTTPExtender object.
func NewHTTPExtender(config *schedulerapi.ExtenderConfig) (algorithm.SchedulerExtender, error) {
	if config.HTTPTimeout.Nanoseconds() == 0 {
		config.HTTPTimeout = time.Duration(DefaultExtenderTimeout)
	}

	transport, err := makeTransport(config)
	if err != nil {
		return nil, err
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   config.HTTPTimeout,
	}
	managedResources := sets.NewString()
	for _, r := range config.ManagedResources {
		managedResources.Insert(string(r.Name))
	}
	return &HTTPExtender{
		extenderURL:      config.URLPrefix,
		preemptVerb:      config.PreemptVerb,
		filterVerb:       config.FilterVerb,
		prioritizeVerb:   config.PrioritizeVerb,
		bindVerb:         config.BindVerb,
		weight:           config.Weight,
		client:           client,
		nodeCacheCapable: config.NodeCacheCapable,
		managedResources: managedResources,
		ignorable:        config.Ignorable,
	}, nil
}

// Name returns extenderURL to identify the extender.
func (h *HTTPExtender) Name() string {
	return h.extenderURL
}

// IsIgnorable returns true indicates scheduling should not fail when this extender
// is unavailable
func (h *HTTPExtender) IsIgnorable() bool {
	return h.ignorable
}

// SupportsPreemption returns true if an extender supports preemption.
// An extender should have preempt verb defined and enabled its own node cache.
func (h *HTTPExtender) SupportsPreemption() bool {
	return len(h.preemptVerb) > 0
}

// ProcessPreemption returns filtered candidate nodes and victims after running preemption logic in extender.
func (h *HTTPExtender) ProcessPreemption(
	pod *v1.Pod,
	nodeToVictims map[*v1.Node]*schedulerapi.Victims,
	nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo,
) (map[*v1.Node]*schedulerapi.Victims, error) {
	var (
		result schedulerapi.ExtenderPreemptionResult
		args   *schedulerapi.ExtenderPreemptionArgs
	)

	if !h.SupportsPreemption() {
		return nil, fmt.Errorf("preempt verb is not defined for extender %v but run into ProcessPreemption", h.extenderURL)
	}

	if h.nodeCacheCapable {
		// If extender has cached node info, pass NodeNameToMetaVictims in args.
		nodeNameToMetaVictims := convertToNodeNameToMetaVictims(nodeToVictims)
		args = &schedulerapi.ExtenderPreemptionArgs{
			Pod:                   pod,
			NodeNameToMetaVictims: nodeNameToMetaVictims,
		}
	} else {
		nodeNameToVictims := convertToNodeNameToVictims(nodeToVictims)
		args = &schedulerapi.ExtenderPreemptionArgs{
			Pod:               pod,
			NodeNameToVictims: nodeNameToVictims,
		}
	}

	if err := h.send(h.preemptVerb, args, &result); err != nil {
		return nil, err
	}

	// Extender will always return NodeNameToMetaVictims.
	// So let's convert it to NodeToVictims by using NodeNameToInfo.
	newNodeToVictims, err := h.convertToNodeToVictims(result.NodeNameToMetaVictims, nodeNameToInfo)
	if err != nil {
		return nil, err
	}
	// Do not override nodeToVictims
	return newNodeToVictims, nil
}

// convertToNodeToVictims converts "nodeNameToMetaVictims" from object identifiers,
// such as UIDs and names, to object pointers.
func (h *HTTPExtender) convertToNodeToVictims(
	nodeNameToMetaVictims map[string]*schedulerapi.MetaVictims,
	nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo,
) (map[*v1.Node]*schedulerapi.Victims, error) {
	nodeToVictims := map[*v1.Node]*schedulerapi.Victims{}
	for nodeName, metaVictims := range nodeNameToMetaVictims {
		victims := &schedulerapi.Victims{
			Pods: []*v1.Pod{},
		}
		for _, metaPod := range metaVictims.Pods {
			pod, err := h.convertPodUIDToPod(metaPod, nodeName, nodeNameToInfo)
			if err != nil {
				return nil, err
			}
			victims.Pods = append(victims.Pods, pod)
		}
		nodeToVictims[nodeNameToInfo[nodeName].Node()] = victims
	}
	return nodeToVictims, nil
}

// convertPodUIDToPod returns v1.Pod object for given MetaPod and node name.
// The v1.Pod object is restored by nodeInfo.Pods().
// It should return error if there's cache inconsistency between default scheduler and extender
// so that this pod or node is missing from nodeNameToInfo.
func (h *HTTPExtender) convertPodUIDToPod(
	metaPod *schedulerapi.MetaPod,
	nodeName string,
	nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo) (*v1.Pod, error) {
	var nodeInfo *schedulernodeinfo.NodeInfo
	if nodeInfo, ok := nodeNameToInfo[nodeName]; ok {
		for _, pod := range nodeInfo.Pods() {
			if string(pod.UID) == metaPod.UID {
				return pod, nil
			}
		}
		return nil, fmt.Errorf("extender: %v claims to preempt pod (UID: %v) on node: %v, but the pod is not found on that node",
			h.extenderURL, metaPod, nodeInfo.Node().Name)
	}

	return nil, fmt.Errorf("extender: %v claims to preempt on node: %v but the node is not found in nodeNameToInfo map",
		h.extenderURL, nodeInfo.Node().Name)
}

// convertToNodeNameToMetaVictims converts from struct type to meta types.
func convertToNodeNameToMetaVictims(
	nodeToVictims map[*v1.Node]*schedulerapi.Victims,
) map[string]*schedulerapi.MetaVictims {
	nodeNameToVictims := map[string]*schedulerapi.MetaVictims{}
	for node, victims := range nodeToVictims {
		metaVictims := &schedulerapi.MetaVictims{
			Pods: []*schedulerapi.MetaPod{},
		}
		for _, pod := range victims.Pods {
			metaPod := &schedulerapi.MetaPod{
				UID: string(pod.UID),
			}
			metaVictims.Pods = append(metaVictims.Pods, metaPod)
		}
		nodeNameToVictims[node.GetName()] = metaVictims
	}
	return nodeNameToVictims
}

// convertToNodeNameToVictims converts from node type to node name as key.
func convertToNodeNameToVictims(
	nodeToVictims map[*v1.Node]*schedulerapi.Victims,
) map[string]*schedulerapi.Victims {
	nodeNameToVictims := map[string]*schedulerapi.Victims{}
	for node, victims := range nodeToVictims {
		nodeNameToVictims[node.GetName()] = victims
	}
	return nodeNameToVictims
}

// Filter based on extender implemented predicate functions. The filtered list is
// expected to be a subset of the supplied list. failedNodesMap optionally contains
// the list of failed nodes and failure reasons.
func (h *HTTPExtender) Filter(
	pod *v1.Pod,
	nodes []*v1.Node, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo,
) ([]*v1.Node, schedulerapi.FailedNodesMap, error) {
	var (
		result     schedulerapi.ExtenderFilterResult
		nodeList   *v1.NodeList
		nodeNames  *[]string
		nodeResult []*v1.Node
		args       *schedulerapi.ExtenderArgs
	)

	if h.filterVerb == "" {
		return nodes, schedulerapi.FailedNodesMap{}, nil
	}

	if h.nodeCacheCapable {
		nodeNameSlice := make([]string, 0, len(nodes))
		for _, node := range nodes {
			nodeNameSlice = append(nodeNameSlice, node.Name)
		}
		nodeNames = &nodeNameSlice
	} else {
		nodeList = &v1.NodeList{}
		for _, node := range nodes {
			nodeList.Items = append(nodeList.Items, *node)
		}
	}

	args = &schedulerapi.ExtenderArgs{
		Pod:       pod,
		Nodes:     nodeList,
		NodeNames: nodeNames,
	}

	if err := h.send(h.filterVerb, args, &result); err != nil {
		return nil, nil, err
	}
	if result.Error != "" {
		return nil, nil, fmt.Errorf(result.Error)
	}

	if h.nodeCacheCapable && result.NodeNames != nil {
		nodeResult = make([]*v1.Node, 0, len(*result.NodeNames))
		for i := range *result.NodeNames {
			nodeResult = append(nodeResult, nodeNameToInfo[(*result.NodeNames)[i]].Node())
		}
	} else if result.Nodes != nil {
		nodeResult = make([]*v1.Node, 0, len(result.Nodes.Items))
		for i := range result.Nodes.Items {
			nodeResult = append(nodeResult, &result.Nodes.Items[i])
		}
	}

	return nodeResult, result.FailedNodes, nil
}

// Prioritize based on extender implemented priority functions. Weight*priority is added
// up for each such priority function. The returned score is added to the score computed
// by Kubernetes scheduler. The total score is used to do the host selection.
func (h *HTTPExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, int, error) {
	var (
		result    schedulerapi.HostPriorityList
		nodeList  *v1.NodeList
		nodeNames *[]string
		args      *schedulerapi.ExtenderArgs
	)

	if h.prioritizeVerb == "" {
		result := schedulerapi.HostPriorityList{}
		for _, node := range nodes {
			result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: 0})
		}
		return &result, 0, nil
	}

	if h.nodeCacheCapable {
		nodeNameSlice := make([]string, 0, len(nodes))
		for _, node := range nodes {
			nodeNameSlice = append(nodeNameSlice, node.Name)
		}
		nodeNames = &nodeNameSlice
	} else {
		nodeList = &v1.NodeList{}
		for _, node := range nodes {
			nodeList.Items = append(nodeList.Items, *node)
		}
	}

	args = &schedulerapi.ExtenderArgs{
		Pod:       pod,
		Nodes:     nodeList,
		NodeNames: nodeNames,
	}

	if err := h.send(h.prioritizeVerb, args, &result); err != nil {
		return nil, 0, err
	}
	return &result, h.weight, nil
}

// Bind delegates the action of binding a pod to a node to the extender.
func (h *HTTPExtender) Bind(binding *v1.Binding) error {
	var result schedulerapi.ExtenderBindingResult
	if !h.IsBinder() {
		// This shouldn't happen as this extender wouldn't have become a Binder.
		return fmt.Errorf("Unexpected empty bindVerb in extender")
	}
	req := &schedulerapi.ExtenderBindingArgs{
		PodName:      binding.Name,
		PodNamespace: binding.Namespace,
		PodUID:       binding.UID,
		Node:         binding.Target.Name,
	}
	if err := h.send(h.bindVerb, &req, &result); err != nil {
		return err
	}
	if result.Error != "" {
		return fmt.Errorf(result.Error)
	}
	return nil
}

// IsBinder returns whether this extender is configured for the Bind method.
func (h *HTTPExtender) IsBinder() bool {
	return h.bindVerb != ""
}

// Helper function to send messages to the extender
func (h *HTTPExtender) send(action string, args interface{}, result interface{}) error {
	out, err := json.Marshal(args)
	if err != nil {
		return err
	}

	url := strings.TrimRight(h.extenderURL, "/") + "/" + action

	req, err := http.NewRequest("POST", url, bytes.NewReader(out))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Failed %v with extender at URL %v, code %v", action, url, resp.StatusCode)
	}

	return json.NewDecoder(resp.Body).Decode(result)
}

// IsInterested returns true if at least one extended resource requested by
// this pod is managed by this extender.
func (h *HTTPExtender) IsInterested(pod *v1.Pod) bool {
	if h.managedResources.Len() == 0 {
		return true
	}
	if h.hasManagedResources(pod.Spec.Containers) {
		return true
	}
	if h.hasManagedResources(pod.Spec.InitContainers) {
		return true
	}
	return false
}

func (h *HTTPExtender) hasManagedResources(containers []v1.Container) bool {
	for i := range containers {
		container := &containers[i]
		for resourceName := range container.Resources.Requests {
			if h.managedResources.Has(string(resourceName)) {
				return true
			}
		}
		for resourceName := range container.Resources.Limits {
			if h.managedResources.Has(string(resourceName)) {
				return true
			}
		}
	}
	return false
}
