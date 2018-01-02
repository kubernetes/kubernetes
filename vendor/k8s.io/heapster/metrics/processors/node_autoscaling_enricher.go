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
	"net/url"
	"time"

	kube_config "k8s.io/heapster/common/kubernetes"
	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/util"
	kube_api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	kube_client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
)

type NodeAutoscalingEnricher struct {
	nodeLister *cache.StoreToNodeLister
	reflector  *cache.Reflector
}

func (this *NodeAutoscalingEnricher) Name() string {
	return "node_autoscaling_enricher"
}

func (this *NodeAutoscalingEnricher) Process(batch *core.DataBatch) (*core.DataBatch, error) {
	nodes, err := this.nodeLister.List()
	if err != nil {
		return nil, err
	}
	for _, node := range nodes.Items {
		if metricSet, found := batch.MetricSets[core.NodeKey(node.Name)]; found {
			metricSet.Labels[core.LabelLabels.Key] = util.LabelsToString(node.Labels, ",")
			availableCpu, _ := node.Status.Capacity[kube_api.ResourceCPU]
			availableMem, _ := node.Status.Capacity[kube_api.ResourceMemory]

			cpuRequested := getInt(metricSet, &core.MetricCpuRequest)
			cpuUsed := getInt(metricSet, &core.MetricCpuUsageRate)
			memRequested := getInt(metricSet, &core.MetricMemoryRequest)
			memUsed := getInt(metricSet, &core.MetricMemoryUsage)

			if availableCpu.MilliValue() != 0 {
				setFloat(metricSet, &core.MetricNodeCpuUtilization, float32(cpuUsed)/float32(availableCpu.MilliValue()))
				setFloat(metricSet, &core.MetricNodeCpuReservation, float32(cpuRequested)/float32(availableCpu.MilliValue()))
				setFloat(metricSet, &core.MetricNodeCpuCapacity, float32(availableCpu.MilliValue()))
			}

			if availableMem.Value() != 0 {
				setFloat(metricSet, &core.MetricNodeMemoryUtilization, float32(memUsed)/float32(availableMem.Value()))
				setFloat(metricSet, &core.MetricNodeMemoryReservation, float32(memRequested)/float32(availableMem.Value()))
				setFloat(metricSet, &core.MetricNodeMemoryCapacity, float32(availableMem.Value()))
			}
		}
	}
	return batch, nil
}

func getInt(metricSet *core.MetricSet, metric *core.Metric) int64 {
	if value, found := metricSet.MetricValues[metric.MetricDescriptor.Name]; found {
		return value.IntValue
	}
	return 0
}

func setFloat(metricSet *core.MetricSet, metric *core.Metric, value float32) {
	metricSet.MetricValues[metric.MetricDescriptor.Name] = core.MetricValue{
		MetricType: core.MetricGauge,
		ValueType:  core.ValueFloat,
		FloatValue: value,
	}
}

func NewNodeAutoscalingEnricher(url *url.URL) (*NodeAutoscalingEnricher, error) {
	kubeConfig, err := kube_config.GetKubeClientConfig(url)
	if err != nil {
		return nil, err
	}
	kubeClient := kube_client.NewOrDie(kubeConfig)

	// watch nodes
	lw := cache.NewListWatchFromClient(kubeClient, "nodes", kube_api.NamespaceAll, fields.Everything())
	nodeLister := &cache.StoreToNodeLister{Store: cache.NewStore(cache.MetaNamespaceKeyFunc)}
	reflector := cache.NewReflector(lw, &kube_api.Node{}, nodeLister.Store, time.Hour)
	reflector.Run()

	return &NodeAutoscalingEnricher{
		nodeLister: nodeLister,
		reflector:  reflector,
	}, nil
}
