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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"io/ioutil"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

type KubeletConfig struct {
	// Following options mostly taken from KubeletConfig, but logic with  KubeletInsecurePort little bit
	// different
	Port uint `json:"port,omitempty"`
	InsecurePort uint `json:"insecurePort,omitempty"`
	HTTPTimeout time.Duration `json:"httpTimeout,omitempty"`
	CertFile string `json:"certFile,omitempty"`
	KeyFile string `json:"keyFile,omitempty"`
	CAFile string `json:"caFile,omitempty"`
}

const (
	// TopologyMatchName is the name of the plugin used in the plugin registry and configurations.
	TopologyMatchName = "TopologyMatch"
)

var _ framework.FilterPlugin = &TopologyMatch{}

type TopologyMatch struct {
	arg KubeletConfig
}


// Name returns name of the plugin. It is used in logs, etc.
func (f *TopologyMatch) Name() string {
	return TopologyMatchName
}

func populateResources(resources *[]v1.ResourceRequirements, containers []v1.Container) {
	for _, container := range containers {
		if len(container.Resources.Limits) == 0 || len(container.Resources.Requests) == 0{
			continue
		}
		*resources = append(*resources, container.Resources)
	}
}

func makeJSONRequest(pod *v1.Pod) ([]byte, error) {
	resources := make([]v1.ResourceRequirements, 0)

	populateResources(&resources, pod.Spec.InitContainers)
	populateResources(&resources, pod.Spec.Containers)

	klog.Infof("resources: %v", resources)
	if len(resources) == 0 {
		return nil, nil
	}
	return json.Marshal(resources)
}

func (t *TopologyMatch) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	// hard coded for test purpose (TODO) get it from config
	//port := 10250
	//certFile := "/etc/kubernetes/pki/apiserver-kubelet-client.crt"
	//keyFile := "/etc/kubernetes/pki/apiserver-kubelet-client.key"
	//caFile := "/etc/kubernetes/pki/ca.crt"
	var rt http.RoundTripper
	var err error

	if t.arg.InsecurePort != 0 {
		config := &kubeletclient.KubeletClientConfig{
				Port:        uint(t.arg.InsecurePort),
				EnableHTTPS: false,
			}
		rt, err = kubeletclient.MakeInsecureTransport(config)
	} else {
		config := &kubeletclient.KubeletClientConfig{
			Port:        uint(t.arg.Port),
			EnableHTTPS: true,
			TLSClientConfig: restclient.TLSClientConfig{
				CertFile: t.arg.CertFile,
				KeyFile: t.arg.KeyFile,
				// TLS Configuration, only applies if EnableHTTPS is true.
				CAFile: t.arg.CAFile,
			},
		}
		rt, err = kubeletclient.MakeTransport(config)
	}


	if err != nil {
		klog.Errorf("Failed to create transport #%v", err)
		return framework.NewStatus(framework.Error, err.Error())
	}
	if rt == nil {
		klog.Error("RoundTripper should not be nil")
		return framework.NewStatus(framework.Error, "RoundTripper is nil")
	}
	topologyResource := fmt.Sprintf("https://%s:%d/ntm/canAssign", nodeInfo.Node().Name, t.arg.Port)
	jsonRequest, err := makeJSONRequest(pod)
	klog.Infof("jsonRequest: %s", jsonRequest)
	if jsonRequest == nil {
		klog.Info("Empty resource list")
		return nil
	}
	//add content type application/json
	req, err := http.NewRequest(http.MethodPost, topologyResource, bytes.NewBuffer(jsonRequest))
	if err != nil {
		klog.Fatal(err)
		return framework.NewStatus(framework.Error, err.Error())
	}
	req.Header.Set("Content-Type", "application/json")
	// todo timeout, keep connection, is it possible
	resp, err := rt.RoundTrip(req)
	if err != nil {
		klog.Fatal(err)
		return framework.NewStatus(framework.Error, err.Error())
	}
	if resp.StatusCode != http.StatusOK {
		dump, err := httputil.DumpResponse(resp, true)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
		}
		return framework.NewStatus(framework.Error, fmt.Sprintf("response: %v", dump))
	}
	answer, err := ioutil.ReadAll(resp.Body)
	klog.Infof("answer: %v, err: %v", string(answer), err)
	var v api.PodAssignVerdict
	err = json.Unmarshal(answer, &v)
	klog.Infof("verdict: %v", v)
	if !v.Ok {
		return framework.NewStatus(framework.Error, fmt.Sprintf("Can't assign pod to node %s, %s, %s", nodeInfo.Node().Name, v.Reason, v.Message))
	}
	return nil
}

// NewTopologyMatch initializes a new plugin and returns it.
func NewTopologyMatch(args runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	kcfg := &KubeletConfig{}
	if err := framework.DecodeInto(args, kcfg); err != nil {
		return nil, err
	}

	topologyMatch := &TopologyMatch{}
	topologyMatch.arg = *kcfg
	return topologyMatch, nil
}
