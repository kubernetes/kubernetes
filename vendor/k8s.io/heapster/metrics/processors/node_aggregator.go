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

// Does not add any nodes.
type NodeAggregator struct {
	MetricsToAggregate []string
}

func (this *NodeAggregator) Name() string {
	return "node_aggregator"
}

func (this *NodeAggregator) Process(batch *core.DataBatch) (*core.DataBatch, error) {
	for key, metricSet := range batch.MetricSets {
		if metricSetType, found := metricSet.Labels[core.LabelMetricSetType.Key]; found && metricSetType == core.MetricSetTypePod {
			// Aggregating pods
			nodeName, found := metricSet.Labels[core.LabelNodename.Key]
			if nodeName == "" {
				glog.V(8).Infof("Skipping pod %s: no node info", key)
				continue
			}
			if found {
				nodeKey := core.NodeKey(nodeName)
				node, found := batch.MetricSets[nodeKey]
				if !found {
					glog.V(1).Info("No metric for node %s, cannot perform node level aggregation.")
				} else if err := aggregate(metricSet, node, this.MetricsToAggregate); err != nil {
					return nil, err
				}
			} else {
				glog.Errorf("No node info in pod %s: %v", key, metricSet.Labels)
			}
		}
	}
	return batch, nil
}
