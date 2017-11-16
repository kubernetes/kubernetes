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

type ApiServerMetrics Metrics

func (m *ApiServerMetrics) Equal(o ApiServerMetrics) bool {
	return (*Metrics)(m).Equal(Metrics(o))
}

func NewApiServerMetrics() ApiServerMetrics {
	result := NewMetrics()
	return ApiServerMetrics(result)
}

func parseApiServerMetrics(data string) (ApiServerMetrics, error) {
	result := NewApiServerMetrics()
	if err := parseMetrics(data, (*Metrics)(&result)); err != nil {
		return ApiServerMetrics{}, err
	}
	return result, nil
}

func (g *MetricsGrabber) getMetricsFromApiServer() (string, error) {
	rawOutput, err := g.client.CoreV1().RESTClient().Get().RequestURI("/metrics").Do().Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
