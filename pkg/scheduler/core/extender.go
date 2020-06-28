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

	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	restclient "k8s.io/client-go/rest"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// DefaultExtenderTimeout defines the default extender timeout in second.
	DefaultExtenderTimeout = 5 * time.Second
)

// HTTPExtender implements the Extender interface.
type HTTPExtender struct {
	extenderURL      string
	preemptVerb      string
	filterVerb       string
	prioritizeVerb   string
	bindVerb         string
	weight           int64
	client           *http.Client
	nodeCacheCapable bool
	managedResources sets.String
	ignorable        bool
}

func makeTransport(config *schedulerapi.Extender) (http.RoundTripper, error) {
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
func NewHTTPExtender(config *schedulerapi.Extender) (framework.Extender, error) {
	if config.HTTPTimeout.Duration.Nanoseconds() == 0 {
		config.HTTPTimeout.Duration = time.Duration(DefaultExtenderTimeout)
	}

	transport, err := makeTransport(config)
	if err != nil {
		return nil, err
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   config.HTTPTimeout.Duration,
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

// Equal is used to check if two extenders are equal
// ignoring the client field, exported for testing
func Equal(e1, e2 *HTTPExtender) bool {
	if e1.extenderURL != e2.extenderURL {
		return false
	}
	if e1.preemptVerb != e2.preemptVerb {
		return false
	}
	if e1.prioritizeVerb != e2.prioritizeVerb {
		return false
	}
	if e1.bindVerb != e2.bindVerb {
		return false
	}
	if e1.weight != e2.weight {
		return false
	}
	if e1.nodeCacheCapable != e2.nodeCacheCapable {
		return false
	}
	if !e1.managedResources.Equal(e2.managedResources) {
		return false
	}
	if e1.ignorable != e2.ignorable {
		return false
	}
	return true
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
	nodeNameToVictims map[string]*extenderv1.Victims,
	nodeInfos framework.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	var (
		result extenderv1.ExtenderPreemptionResult
		args   *extenderv1.ExtenderPreemptionArgs
	)

	if !h.SupportsPreemption() {
		return nil, fmt.Errorf("preempt verb is not defined for extender %v but run into ProcessPreemption", h.extenderURL)
	}

	if h.nodeCacheCapable {
		// If extender has cached node info, pass NodeNameToMetaVictims in args.
		nodeNameToMetaVictims := convertToNodeNameToMetaVictims(nodeNameToVictims)
		args = &extenderv1.ExtenderPreemptionArgs{
			Pod:                   pod,
			NodeNameToMetaVictims: nodeNameToMetaVictims,
		}
	} else {
		args = &extenderv1.ExtenderPreemptionArgs{
			Pod:               pod,
			NodeNameToVictims: nodeNameToVictims,
		}
	}

	if err := h.send(h.preemptVerb, args, &result); err != nil {
		return nil, err
	}

	// Extender will always return NodeNameToMetaVictims.
	// So let's convert it to NodeNameToVictims by using <nodeInfos>.
	newNodeNameToVictims, err := h.convertToNodeNameToVictims(result.NodeNameToMetaVictims, nodeInfos)
	if err != nil {
		return nil, err
	}
	// Do not override <nodeNameToVictims>.
	return newNodeNameToVictims, nil
}

// convertToNodeNameToVictims converts "nodeNameToMetaVictims" from object identifiers,
// such as UIDs and names, to object pointers.
func (h *HTTPExtender) convertToNodeNameToVictims(
	nodeNameToMetaVictims map[string]*extenderv1.MetaVictims,
	nodeInfos framework.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	nodeNameToVictims := map[string]*extenderv1.Victims{}
	for nodeName, metaVictims := range nodeNameToMetaVictims {
		nodeInfo, err := nodeInfos.Get(nodeName)
		if err != nil {
			return nil, err
		}
		victims := &extenderv1.Victims{
			Pods: []*v1.Pod{},
		}
		for _, metaPod := range metaVictims.Pods {
			pod, err := h.convertPodUIDToPod(metaPod, nodeInfo)
			if err != nil {
				return nil, err
			}
			victims.Pods = append(victims.Pods, pod)
		}
		nodeNameToVictims[nodeName] = victims
	}
	return nodeNameToVictims, nil
}

// convertPodUIDToPod returns v1.Pod object for given MetaPod and node info.
// The v1.Pod object is restored by nodeInfo.Pods().
// It returns an error if there's cache inconsistency between default scheduler
// and extender, i.e. when the pod is not found in nodeInfo.Pods.
func (h *HTTPExtender) convertPodUIDToPod(
	metaPod *extenderv1.MetaPod,
	nodeInfo *framework.NodeInfo) (*v1.Pod, error) {
	for _, p := range nodeInfo.Pods {
		if string(p.Pod.UID) == metaPod.UID {
			return p.Pod, nil
		}
	}
	return nil, fmt.Errorf("extender: %v claims to preempt pod (UID: %v) on node: %v, but the pod is not found on that node",
		h.extenderURL, metaPod, nodeInfo.Node().Name)
}

// convertToNodeNameToMetaVictims converts from struct type to meta types.
func convertToNodeNameToMetaVictims(
	nodeNameToVictims map[string]*extenderv1.Victims,
) map[string]*extenderv1.MetaVictims {
	nodeNameToMetaVictims := map[string]*extenderv1.MetaVictims{}
	for node, victims := range nodeNameToVictims {
		metaVictims := &extenderv1.MetaVictims{
			Pods: []*extenderv1.MetaPod{},
		}
		for _, pod := range victims.Pods {
			metaPod := &extenderv1.MetaPod{
				UID: string(pod.UID),
			}
			metaVictims.Pods = append(metaVictims.Pods, metaPod)
		}
		nodeNameToMetaVictims[node] = metaVictims
	}
	return nodeNameToMetaVictims
}

// Filter based on extender implemented predicate functions. The filtered list is
// expected to be a subset of the supplied list; otherwise the function returns an error.
// failedNodesMap optionally contains the list of failed nodes and failure reasons.
func (h *HTTPExtender) Filter(
	pod *v1.Pod,
	nodes []*v1.Node,
) ([]*v1.Node, extenderv1.FailedNodesMap, error) {
	var (
		result     extenderv1.ExtenderFilterResult
		nodeList   *v1.NodeList
		nodeNames  *[]string
		nodeResult []*v1.Node
		args       *extenderv1.ExtenderArgs
	)
	fromNodeName := make(map[string]*v1.Node)
	for _, n := range nodes {
		fromNodeName[n.Name] = n
	}

	if h.filterVerb == "" {
		return nodes, extenderv1.FailedNodesMap{}, nil
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

	args = &extenderv1.ExtenderArgs{
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
		nodeResult = make([]*v1.Node, len(*result.NodeNames))
		for i, nodeName := range *result.NodeNames {
			if n, ok := fromNodeName[nodeName]; ok {
				nodeResult[i] = n
			} else {
				return nil, nil, fmt.Errorf(
					"extender %q claims a filtered node %q which is not found in the input node list",
					h.extenderURL, nodeName)
			}
		}
	} else if result.Nodes != nil {
		nodeResult = make([]*v1.Node, len(result.Nodes.Items))
		for i := range result.Nodes.Items {
			nodeResult[i] = &result.Nodes.Items[i]
		}
	}

	return nodeResult, result.FailedNodes, nil
}

// Prioritize based on extender implemented priority functions. Weight*priority is added
// up for each such priority function. The returned score is added to the score computed
// by Kubernetes scheduler. The total score is used to do the host selection.
func (h *HTTPExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*extenderv1.HostPriorityList, int64, error) {
	var (
		result    extenderv1.HostPriorityList
		nodeList  *v1.NodeList
		nodeNames *[]string
		args      *extenderv1.ExtenderArgs
	)

	if h.prioritizeVerb == "" {
		result := extenderv1.HostPriorityList{}
		for _, node := range nodes {
			result = append(result, extenderv1.HostPriority{Host: node.Name, Score: 0})
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

	args = &extenderv1.ExtenderArgs{
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
	var result extenderv1.ExtenderBindingResult
	if !h.IsBinder() {
		// This shouldn't happen as this extender wouldn't have become a Binder.
		return fmt.Errorf("Unexpected empty bindVerb in extender")
	}
	req := &extenderv1.ExtenderBindingArgs{
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
