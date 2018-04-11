/*
Copyright 2017 The Kubernetes Authors.

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

package priorities

import (
	"fmt"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	resourceclient "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1"
)

// Note: Least used based on metrics priority function ensures that the node which has least current cpu utilization
// gets pods. We use metrics-server for getting the utilization information.

// usageDataOnNode contains the resource metrics client used to get node info.
type usageDataOnNode struct {
	metricsNodeClient priorityutil.MetricsClient
}

// NewLeastUsedNodeBasedOnMetrics returns leastUsagePriorityMap needed.
func NewLeastUsedNodeBasedOnMetrics(metricsClient *resourceclient.MetricsV1beta1Client) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
	metricsNodeClient := priorityutil.NewRESTMetricsClient(metricsClient)
	currentUsage := &usageDataOnNode{
		metricsNodeClient: metricsNodeClient,
	}
	return currentUsage.leastUsagePriorityMap, nil
}

// getScore checks if there is any other node which has less cpu utilization when compared to current node. If there is,
// the score for current node becomes 0 or else it returns 10 meaning current node is least utilized node in the cluster.
func getScore(currentNodeCPUUtil int64, nodeUtilInfo priorityutil.NodeMetricsInfo) int {
	for _, info := range nodeUtilInfo {
		if info < currentNodeCPUUtil {
			// There is a node which has utilization less than current one. So, this node gets a score of 0.
			return int(0)
		}
	}
	return int(10)
}

// LeastUsagePriorityMap is a priority function that finds the node with least utilization in the cluster. We rely on
// metrics-server to get information related to all the nodes in the cluster. Following are possible scenarios
// - If there is a node in the cluster which has less less utilization than the current node, it gets a 0.
// - Else, the current node has the least utilization in the cluster and it gets a score of 10.
func (u *usageDataOnNode) leastUsagePriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("Node not found")
	}
	var nodeUtilInfo = make(priorityutil.NodeMetricsInfo)
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		nodeUtilInfo = priorityMeta.nodeUtilInfo

	} else {
		glog.V(5).Infof("Falling back to computing usage data from metrics")
		nodeUtilInfo = getNodeUtilizationInfo(u.metricsNodeClient)
	}
	if len(nodeUtilInfo) == 0 {
		// It's ok even if we don't get the node utilization score. There is a very good chance metrics-server is warming up
		// So let us return a score of 0 for every node.
		return schedulerapi.HostPriority{Host: node.Name, Score: int(0)}, nil
	}
	currentNodeCPUUtil := nodeUtilInfo[node.Name]
	score := getScore(currentNodeCPUUtil, nodeUtilInfo)
	glog.V(5).Infof("The least used node based on metrics score is %v for the node %v for pod %v", score, node.Name, pod.Name)
	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: score,
	}, nil
}
