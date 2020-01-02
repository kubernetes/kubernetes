/*
Copyright 2016 The Kubernetes Authors.

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
	"sort"

	v1 "k8s.io/api/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
)

func runMapReducePriority(mapFn PriorityMapFunction, reduceFn PriorityReduceFunction, metaData interface{}, pod *v1.Pod, sharedLister schedulerlisters.SharedLister, nodes []*v1.Node) (framework.NodeScoreList, error) {
	result := make(framework.NodeScoreList, 0, len(nodes))
	for i := range nodes {
		nodeInfo, err := sharedLister.NodeInfos().Get(nodes[i].Name)
		if err != nil {
			return nil, err
		}
		hostResult, err := mapFn(pod, metaData, nodeInfo)
		if err != nil {
			return nil, err
		}
		result = append(result, hostResult)
	}
	if reduceFn != nil {
		if err := reduceFn(pod, metaData, sharedLister, result); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func sortNodeScoreList(out framework.NodeScoreList) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Name < out[j].Name
		}
		return out[i].Score < out[j].Score
	})
}
