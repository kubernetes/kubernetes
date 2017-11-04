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

package metrics

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

type KubeletMetrics Metrics

func (m *KubeletMetrics) Equal(o KubeletMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewKubeletMetrics() KubeletMetrics {
	result := NewMetrics()
	return KubeletMetrics(result)
}

// GrabKubeletMetricsWithoutProxy retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func GrabKubeletMetricsWithoutProxy(nodeName string) (KubeletMetrics, error) {
	metricsEndpoint := "http://%s/metrics"
	resp, err := http.Get(fmt.Sprintf(metricsEndpoint, nodeName))
	if err != nil {
		return KubeletMetrics{}, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(string(body))
}

func parseKubeletMetrics(data string) (KubeletMetrics, error) {
	result := NewKubeletMetrics()
	if err := parseMetrics(data, (*Metrics)(&result)); err != nil {
		return KubeletMetrics{}, err
	}
	return result, nil
}

func (g *MetricsGrabber) getMetricsFromNode(nodeName string, kubeletPort int) (string, error) {
	// There's a problem with timing out during proxy. Wrapping this in a goroutine to prevent deadlock.
	// Hanging goroutine will be leaked.
	finished := make(chan struct{})
	var err error
	var rawOutput []byte
	go func() {
		rawOutput, err = g.client.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, kubeletPort)).
			Suffix("metrics").
			Do().Raw()
		finished <- struct{}{}
	}()
	select {
	case <-time.After(ProxyTimeout):
		return "", fmt.Errorf("Timed out when waiting for proxy to gather metrics from %v", nodeName)
	case <-finished:
		if err != nil {
			return "", err
		}
		return string(rawOutput), nil
	}
}
