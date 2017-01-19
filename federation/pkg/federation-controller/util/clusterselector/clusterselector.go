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

package clusterselector

import (
	"encoding/json"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	federation_v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
)

// Parses the cluster selector annotation to find out if the object with that annotation should be forwarded to a cluster with the given clusterLabels.
func SendToCluster(clusterLabels map[string]string, annotations map[string]string) (bool, error) {
	// Check if there is an annotation for ClusterSelector
	if val, ok := annotations[federation_v1beta1.FederationClusterSelector]; ok {
		// Check if the Annotation contains valid data
		selector := labels.NewSelector()
		requirements := make([]federation_v1beta1.ClusterSelectorRequirement, 0)
		if err := json.Unmarshal([]byte(val), &requirements); err == nil {
			for _, requirement := range requirements {
				r, err := labels.NewRequirement(requirement.Key, ConvertOperator(requirement.Operator), requirement.Values)
				if err != nil {
					// Stop processing, assume failure and throw an error since we have no way of knowing the end users intent for this or any other clusters.
					glog.V(2).Infof("Unable to convert ClusterSelector Annotation to Requirement: %+v, %s", requirement, err)
					return false, err
				}
				selector = selector.Add(*r)
			}
			if !selector.Matches(labels.Set(clusterLabels)) {
				glog.V(8).Infof("Selector: %+v does not match labels: %+v", selector.String(), clusterLabels)
				return false, nil
			}
		} else {
			// Stop processing, assume failure and throw an error since we have no way of knowing the end users intent for this or any other clusters.
			glog.V(2).Infof("Unable to parse ClusterSelector Annotation: %s", err)
			return false, err
		}
	}
	return true, nil
}

// ConvertOperator converts a string operator into selection.Operator type
func ConvertOperator(source string) selection.Operator {
	var op selection.Operator
	switch source {
	case "!", "DoesNotExist":
		op = selection.DoesNotExist
	case "=":
		op = selection.Equals
	case "==":
		op = selection.DoubleEquals
	case "in", "In":
		op = selection.In
	case "!=":
		op = selection.NotEquals
	case "notin", "NotIn":
		op = selection.NotIn
	case "exists", "Exists":
		op = selection.Exists
	case "gt", "Gt":
		op = selection.GreaterThan
	case "lt", "Lt":
		op = selection.LessThan
	}
	return op
}
