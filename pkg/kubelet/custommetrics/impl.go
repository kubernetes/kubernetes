/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package custommetrics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// Key for the annotation defining custom metrics for pod.
	// More info: http://releases.k8s.io/HEAD/docs/proposals/custom-metrics.md#api
	CustomMetricsAnnotationKey = "metrics.alpha.kubernetes.io/custom-endpoints"

	CustomMetricsDefinitionContainerFile = "cadvisor-custom-metrics-prometheus.json"

	CustomMetricsDefinitionContainerPath = "/etc/" + CustomMetricsDefinitionContainerFile
)

// CustomMetricsEndpoint is a structure defining endpoint for pod's custom metric.
// It is parsed from JSON from the value of the annotation with CustomMetricsAnnotationKey.
// More info: http://releases.k8s.io/HEAD/docs/proposals/custom-metrics.md#api
type CustomMetricsEndpoint struct {
	API   string   `json:"api",omitempty`
	Path  string   `json:"path",omitempty`
	Port  int      `json:"port"`
	Names []string `json:"names"`
}

func MakeCustomMetricsMountIfRequired(
	pod *api.Pod, container *api.Container, podContainerDir string, maxCustomMetricsPerPod int) (*kubecontainer.Mount, error) {
	if definition, ok := pod.Annotations[CustomMetricsAnnotationKey]; ok {
		customMetricsMount, err := makeCustomMetricsMount(
			container, podContainerDir, pod.Status.PodIP, definition, maxCustomMetricsPerPod)
		if err != nil {
			return nil, err
		}
		return customMetricsMount, nil
	}
	return nil, nil
}

func VerifyCustomMetricsAnnotation(pod *api.Pod, maxCustomMetricsPerPod int) error {
	if definition, ok := pod.Annotations[CustomMetricsAnnotationKey]; ok {
		for _, container := range pod.Spec.Containers {
			if _, err := getCustomMetricsEndpointsForContainer(&container, definition, maxCustomMetricsPerPod); err != nil {
				return err
			}
		}
	}
	return nil
}

func makeCustomMetricsMount(container *api.Container, podContainerDir, podIP, definition string, maxCustomMetricsPerPod int) (*kubecontainer.Mount, error) {
	endpoints, err := getCustomMetricsEndpointsForContainer(container, definition, maxCustomMetricsPerPod)
	if err != nil {
		return nil, err
	}
	if len(endpoints) == 0 {
		return nil, nil
	}

	if err := os.MkdirAll(podContainerDir, 0750); err != nil {
		return nil, fmt.Errorf("error creating pod-container directory: %v", err)
	}
	hostsFilePath := path.Join(podContainerDir, CustomMetricsDefinitionContainerFile)
	if err := ensureCustomMetricsFile(hostsFilePath, podIP, endpoints); err != nil {
		return nil, err
	}
	return &kubecontainer.Mount{
		Name:          "k8s-custom-metrics",
		ContainerPath: CustomMetricsDefinitionContainerPath,
		HostPath:      hostsFilePath,
		ReadOnly:      false,
	}, nil
}

func getCustomMetricsEndpointsForContainer(container *api.Container, definition string, maxCustomMetricsPerPod int) ([]CustomMetricsEndpoint, error) {
	var endpoints []CustomMetricsEndpoint
	err := json.Unmarshal([]byte(definition), &endpoints)
	if err != nil {
		return nil, fmt.Errorf("error while parsing custom metrics definition %q: %v", definition, err)
	}

	numCustomMetrics := 0
	for _, endpoint := range endpoints {
		if len(endpoint.Names) == 0 {
			return nil, fmt.Errorf("custom metrics endpoint definition cannot contain empty names: +%v", endpoint)
		}
		numCustomMetrics += len(endpoint.Names)
	}
	if numCustomMetrics > maxCustomMetricsPerPod {
		return nil, fmt.Errorf("too many custom metrics per pod, only %d allowed: %v", maxCustomMetricsPerPod, endpoints)
	}

	var matching []CustomMetricsEndpoint
	for _, port := range container.Ports {
		for _, endpoint := range endpoints {
			if port.ContainerPort == endpoint.Port {
				matching = append(matching, endpoint)
			}
		}
	}
	return matching, nil
}

func ensureCustomMetricsFile(fileName, hostIP string, endpoints []CustomMetricsEndpoint) error {
	if len(endpoints) > 1 {
		return fmt.Errorf("too many endpoints (only one endpoint per container is allowed)")
	}
	endpoint := endpoints[0]
	if len(endpoint.Names) == 0 {
		return fmt.Errorf("names of custom metrics to collect cannot be empty, endpoint: +%v", endpoint)
	}

	if _, err := os.Stat(fileName); os.IsExist(err) {
		glog.V(4).Infof("kubernetes-managed metrics file exits. Will not be recreated: %q", fileName)
		return nil
	}
	var buffer bytes.Buffer
	buffer.WriteString("{\n")
	buffer.WriteString(fmt.Sprintf("  \"endpoint\" : \"http://%s:%d%s\",\n", hostIP, endpoint.Port, endpoint.Path))
	buffer.WriteString("  \"metrics_config\" : [\n")
	for i, name := range endpoint.Names {
		comma := ","
		if i == len(endpoint.Names)-1 { // Do not print comma after the last name.
			comma = ""
		}
		buffer.WriteString(fmt.Sprintf("    %q%s\n", name, comma))
	}
	buffer.WriteString("  ]\n")
	buffer.WriteString("}\n")
	return ioutil.WriteFile(fileName, buffer.Bytes(), 0644)
}
