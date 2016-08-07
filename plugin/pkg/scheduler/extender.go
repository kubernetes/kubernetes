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

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

const (
	DefaultExtenderTimeout = 5 * time.Second
)

// HTTPExtender implements the algorithm.SchedulerExtender interface.
type HTTPExtender struct {
	extenderURL    string
	filterVerb     string
	prioritizeVerb string
	weight         int
	apiVersion     string
	client         *http.Client
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

func NewHTTPExtender(config *schedulerapi.ExtenderConfig, apiVersion string) (algorithm.SchedulerExtender, error) {
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
		extenderURL:    config.URLPrefix,
		apiVersion:     apiVersion,
		filterVerb:     config.FilterVerb,
		prioritizeVerb: config.PrioritizeVerb,
		weight:         config.Weight,
		client:         client,
	}, nil
}

// Filter based on extender implemented predicate functions. The filtered list is
// expected to be a subset of the supplied list. failedNodesMap optionally contains
// the list of failed nodes and failure reasons.
func (h *HTTPExtender) Filter(pod *api.Pod, nodes []*api.Node) ([]*api.Node, schedulerapi.FailedNodesMap, error) {
	var result schedulerapi.ExtenderFilterResult

	if h.filterVerb == "" {
		return nodes, schedulerapi.FailedNodesMap{}, nil
	}

	nodeItems := make([]api.Node, 0, len(nodes))
	for _, node := range nodes {
		nodeItems = append(nodeItems, *node)
	}
	args := schedulerapi.ExtenderArgs{
		Pod:   *pod,
		Nodes: api.NodeList{Items: nodeItems},
	}

	if err := h.send(h.filterVerb, &args, &result); err != nil {
		return nil, nil, err
	}
	if result.Error != "" {
		return nil, nil, fmt.Errorf(result.Error)
	}

	nodeResult := make([]*api.Node, 0, len(result.Nodes.Items))
	for i := range result.Nodes.Items {
		nodeResult = append(nodeResult, &result.Nodes.Items[i])
	}
	return nodeResult, result.FailedNodes, nil
}

// Prioritize based on extender implemented priority functions. Weight*priority is added
// up for each such priority function. The returned score is added to the score computed
// by Kubernetes scheduler. The total score is used to do the host selection.
func (h *HTTPExtender) Prioritize(pod *api.Pod, nodes []*api.Node) (*schedulerapi.HostPriorityList, int, error) {
	var result schedulerapi.HostPriorityList

	if h.prioritizeVerb == "" {
		result := schedulerapi.HostPriorityList{}
		for _, node := range nodes {
			result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: 0})
		}
		return &result, 0, nil
	}

	nodeItems := make([]api.Node, 0, len(nodes))
	for _, node := range nodes {
		nodeItems = append(nodeItems, *node)
	}
	args := schedulerapi.ExtenderArgs{
		Pod:   *pod,
		Nodes: api.NodeList{Items: nodeItems},
	}

	if err := h.send(h.prioritizeVerb, &args, &result); err != nil {
		return nil, 0, err
	}
	return &result, h.weight, nil
}

// Helper function to send messages to the extender
func (h *HTTPExtender) send(action string, args interface{}, result interface{}) error {
	out, err := json.Marshal(args)
	if err != nil {
		return err
	}

	url := h.extenderURL + "/" + h.apiVersion + "/" + action

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
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(body, result); err != nil {
		return err
	}
	return nil
}
