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
	"context"

	"k8s.io/component-base/metrics/testutil"
)

// APIServerMetrics is metrics for API server
type APIServerMetrics testutil.Metrics

// Equal returns true if all metrics are the same as the arguments.
func (m *APIServerMetrics) Equal(o APIServerMetrics) bool {
	return (*testutil.Metrics)(m).Equal(testutil.Metrics(o))
}

func newAPIServerMetrics() APIServerMetrics {
	result := testutil.NewMetrics()
	return APIServerMetrics(result)
}

func parseAPIServerMetrics(data string) (APIServerMetrics, error) {
	result := newAPIServerMetrics()
	if err := testutil.ParseMetrics(data, (*testutil.Metrics)(&result)); err != nil {
		return APIServerMetrics{}, err
	}
	return result, nil
}

func (g *Grabber) getMetricsFromAPIServer() (string, error) {
	rawOutput, err := g.client.CoreV1().RESTClient().Get().RequestURI("/metrics").Do(context.TODO()).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
