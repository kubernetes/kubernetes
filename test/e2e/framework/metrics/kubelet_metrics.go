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

const (
	proxyTimeout = 2 * time.Minute
)

// KubeletMetrics is metrics for kubelet
type KubeletMetrics Metrics

// Equal returns true if all metrics are the same as the arguments.
func (m *KubeletMetrics) Equal(o KubeletMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

// NewKubeletMetrics returns new metrics which are initialized.
func NewKubeletMetrics() KubeletMetrics {
	result := NewMetrics()
	return KubeletMetrics(result)
}

// GrabKubeletMetricsWithoutProxy retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func GrabKubeletMetricsWithoutProxy(nodeName, path string) (KubeletMetrics, error) {
	resp, err := http.Get(fmt.Sprintf("http://%s%s", nodeName, path))
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

func (g *Grabber) getMetricsFromNode(nodeName string, kubeletPort int) (string, error) {
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
	case <-time.After(proxyTimeout):
		return "", fmt.Errorf("Timed out when waiting for proxy to gather metrics from %v", nodeName)
	case <-finished:
		if err != nil {
			return "", err
		}
		return string(rawOutput), nil
	}
}
