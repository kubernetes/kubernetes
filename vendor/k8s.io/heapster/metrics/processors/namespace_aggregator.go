// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package processors

import (
	"github.com/golang/glog"
	"k8s.io/heapster/metrics/core"
)

type NamespaceAggregator struct {
	MetricsToAggregate []string
}

func (this *NamespaceAggregator) Name() string {
	return "namespace_aggregator"
}

func (this *NamespaceAggregator) Process(batch *core.DataBatch) (*core.DataBatch, error) {
	namespaces := make(map[string]*core.MetricSet)
	for key, metricSet := range batch.MetricSets {
		if metricSetType, found := metricSet.Labels[core.LabelMetricSetType.Key]; found && metricSetType == core.MetricSetTypePod {
			// Aggregating pods
			if namespaceName, found := metricSet.Labels[core.LabelNamespaceName.Key]; found {
				namespaceKey := core.NamespaceKey(namespaceName)
				namespace, found := namespaces[namespaceKey]
				if !found {
					if nsFromBatch, found := batch.MetricSets[namespaceKey]; found {
						namespace = nsFromBatch
					} else {
						namespace = namespaceMetricSet(namespaceName, metricSet.Labels[core.LabelPodNamespaceUID.Key])
						namespaces[namespaceKey] = namespace
					}
				}
				if err := aggregate(metricSet, namespace, this.MetricsToAggregate); err != nil {
					return nil, err
				}
			} else {
				glog.Errorf("No namespace info in pod %s: %v", key, metricSet.Labels)
			}
		}
	}
	for key, val := range namespaces {
		batch.MetricSets[key] = val
	}
	return batch, nil
}

func namespaceMetricSet(namespaceName, uid string) *core.MetricSet {
	return &core.MetricSet{
		MetricValues: make(map[string]core.MetricValue),
		Labels: map[string]string{
			core.LabelMetricSetType.Key:   core.MetricSetTypeNamespace,
			core.LabelNamespaceName.Key:   namespaceName,
			core.LabelPodNamespaceUID.Key: uid,
		},
	}
}
