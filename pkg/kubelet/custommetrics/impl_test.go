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
	"io/ioutil"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestMakeCustomMetricsFailure(t *testing.T) {
	type FailureTestCase struct {
		definition string
		error      string
	}

	testCases := []FailureTestCase{
		{"wrong json", "invalid character 'w' looking for beginning of value"},
		{"[ { \"path\": \"/metrics\", \"port\": 8080 } ]", "custom metrics endpoint definition cannot contain empty names"},
		{"[ { \"path\": \"/metrics\", \"port\": 8080, \"names\": [\"a\", \"b\", \"c\", \"d\", \"e\", \"f\"] } ]", "too many custom metrics per pod"},
	}

	// test makeCustomMetricsMount function
	for _, tc := range testCases {
		_, err := makeCustomMetricsMount(&api.Container{}, "/tmp", "172.3.4.12", tc.definition, 5)
		if !strings.Contains(err.Error(), tc.error) {
			t.Errorf("Unexpected error during mount, got: %v, expected %s", err, tc.error)
		}
	}

	// test VerifyCustomMetricsAnnotation function
	for _, tc := range testCases {
		pod := api.Pod{}
		pod.Spec = api.PodSpec{Containers: []api.Container{{}}}
		pod.Annotations = map[string]string{}
		pod.Annotations[CustomMetricsAnnotationKey] = tc.definition
		err := VerifyCustomMetricsAnnotation(&pod, 5)
		if !strings.Contains(err.Error(), tc.error) {
			t.Errorf("Unexpected error during verify, got: %v, expected %s", err, tc.error)
		}
	}
}

func TestMakeCustomMetricsMountSuccess(t *testing.T) {
	type SuccessTestCase struct {
		definition string
		golden     string
	}

	testCases := []SuccessTestCase{
		{
			"[ { \"path\": \"/metrics\", \"port\": 8080, \"names\": [\"tralala\"] } ]",
			"{\n  \"endpoint\" : \"http://172.3.4.12:8080/metrics\",\n  \"metrics_config\" : [\n    \"tralala\"\n  ]\n}\n",
		},
		{
			"[ { \"path\": \"/metrics\", \"port\": 8070, \"names\": [\"a\", \"b\"] }, { \"path\": \"/metrics\", \"port\": 8080, \"names\": [\"c\", \"d\"] } ]",
			"{\n  \"endpoint\" : \"http://172.3.4.12:8080/metrics\",\n  \"metrics_config\" : [\n    \"c\",\n    \"d\"\n  ]\n}\n",
		},
	}

	container := api.Container{
		Ports: []api.ContainerPort{
			{
				ContainerPort: 8080,
			},
		},
	}
	metricsFilePath := "/tmp/cadvisor-custom-metrics-prometheus.json"
	expectedMount := kubecontainer.Mount{
		"k8s-custom-metrics",
		CustomMetricsDefinitionContainerPath,
		metricsFilePath,
		false,
		false,
	}

	for _, tc := range testCases {
		mount, err := makeCustomMetricsMount(&container, "/tmp", "172.3.4.12", tc.definition, 5)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(*mount, expectedMount) {
			t.Errorf("Unexpected mounts: Expected %#v got %#v.  Container was: %#v", expectedMount, *mount, container)
		}
		metricsFile, err := ioutil.ReadFile(metricsFilePath)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if string(metricsFile) != tc.golden {
			t.Errorf("Unexpected metrics file, expected:\n%s, got:\n%s", tc.golden, metricsFile)
		}
	}
}
