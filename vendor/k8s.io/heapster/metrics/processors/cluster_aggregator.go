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

import "k8s.io/heapster/metrics/core"

type ClusterAggregator struct {
	MetricsToAggregate []string
}

func (this *ClusterAggregator) Name() string {
	return "cluster_aggregator"
}

func (this *ClusterAggregator) Process(batch *core.DataBatch) (*core.DataBatch, error) {
	clusterKey := core.ClusterKey()
	cluster := clusterMetricSet()
	for _, metricSet := range batch.MetricSets {
		if metricSetType, found := metricSet.Labels[core.LabelMetricSetType.Key]; found &&
			(metricSetType == core.MetricSetTypeNamespace || metricSetType == core.MetricSetTypeSystemContainer) {
			if err := aggregate(metricSet, cluster, this.MetricsToAggregate); err != nil {
				return nil, err
			}
		}
	}
	batch.MetricSets[clusterKey] = cluster
	return batch, nil
}

func clusterMetricSet() *core.MetricSet {
	return &core.MetricSet{
		MetricValues: make(map[string]core.MetricValue),
		Labels: map[string]string{
			core.LabelMetricSetType.Key: core.MetricSetTypeCluster,
		},
	}
}
