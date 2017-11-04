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
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

const (
	DefaultExtenderTimeout = 5 * time.Second
)

// HTTPExtender implements the algorithm.SchedulerExtender interface.
type HTTPExtender struct {
	extenderURL      string
	filterVerb       string
	prioritizeVerb   string
	bindVerb         string
	weight           int
	client           *http.Client
	nodeCacheCapable bool
}

func makeTransport(config *schedulerapi.ExtenderConfig) (http.RoundTripper, error) {
	var cfg restclient.Config
	if config.TLSConfig != nil {
		cfg.TLSClientConfig = *config.TLSConfig
	}
	if config.EnableHttps {
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
	return &HTTPExtender{
		extenderURL:      config.URLPrefix,
		filterVerb:       config.FilterVerb,
		prioritizeVerb:   config.PrioritizeVerb,
		bindVerb:         config.BindVerb,
		weight:           config.Weight,
		client:           client,
		nodeCacheCapable: config.NodeCacheCapable,
	}, nil
}

// Filter based on extender implemented predicate functions. The filtered list is
// expected to be a subset of the supplied list. failedNodesMap optionally contains
// the list of failed nodes and failure reasons.
func (h *HTTPExtender) Filter(pod *v1.Pod, nodes []*v1.Node, nodeNameToInfo map[string]*schedulercache.NodeInfo) ([]*v1.Node, schedulerapi.FailedNodesMap, error) {
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
		Pod:       *pod,
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
		Pod:       *pod,
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
		return fmt.Errorf("Failed %v with extender at URL %v, code %v", action, h.extenderURL, resp.StatusCode)
	}

	return json.NewDecoder(resp.Body).Decode(result)
}
