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
	"encoding/json"
	"fmt"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"time"

	metricsapi "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
)

// PodResourceInfo contains pod resourcemetric values as a map from pod names to
// metric values
type PodResourceInfo map[string]int64

// PodMetricsInfo contains pod resourcemetric values as a map from pod names to
// metric values
type PodMetricsInfo map[string]float64

// MetricsClient knows how to query a remote interface to retrieve container-level
// resource metrics as well as pod-level arbitrary metrics
type MetricsClient interface {
	// GetClusterAverageResourceMetrics gets the given resource metric
	// (and an associated oldest timestamp) for all pods of the given cluster
	// matching the specified selector in the given namespace
	GetClusterResourceMetrics(clusterclient clientset.Interface, resource v1.ResourceName, namespace string, selector labels.Selector) (PodResourceInfo, time.Time, error)
}

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

type HeapsterMetricsClient struct {
	heapsterNamespace string
	heapsterScheme    string
	heapsterService   string
	heapsterPort      string
}

//NewHeapsterMetricsClient gets a metrics client for a given federated cluster
func NewHeapsterMetricsClient(namespace, scheme, service, port string) MetricsClient {
	return &HeapsterMetricsClient{
		heapsterNamespace: namespace,
		heapsterScheme:    scheme,
		heapsterService:   service,
		heapsterPort:      port,
	}
}

func (h *HeapsterMetricsClient) GetClusterResourceMetrics(clusterclient clientset.Interface, resource v1.ResourceName, namespace string, selector labels.Selector) (PodResourceInfo, time.Time, error) {
	metricPath := fmt.Sprintf("/apis/metrics/v1alpha1/namespaces/%s/pods", namespace)
	params := map[string]string{"labelSelector": selector.String()}

	resultRaw, err := clusterclient.Core().Services(h.heapsterNamespace).
		ProxyGet(h.heapsterScheme, h.heapsterService, h.heapsterPort, metricPath, params).
		DoRaw()
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get heapster service: %v", err)
	}

	glog.V(4).Infof("Heapster metrics result: %s", string(resultRaw))

	metrics := metricsapi.PodMetricsList{}
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to unmarshal heapster response: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from heapster")
	}

	res := make(PodResourceInfo, len(metrics.Items))

	for _, m := range metrics.Items {
		podSum := int64(0)
		missing := len(m.Containers) == 0
		for _, c := range m.Containers {
			resValue, found := c.Usage[v1.ResourceName(resource)]
			if !found {
				missing = true
				glog.V(2).Infof("missing resource metric %v for container %s in pod %s/%s", resource, c.Name, namespace, m.Name)
				continue
			}
			podSum += resValue.MilliValue()
		}

		if !missing {
			res[m.Name] = int64(podSum)
		}
	}

	timestamp := metrics.Items[0].Timestamp.Time

	return res, timestamp, nil
}
