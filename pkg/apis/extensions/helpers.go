/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package extensions

import (
	"fmt"

	"sort"

	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

// PodSelectorAsSelector converts the PodSelector api type into a struct that implements
// labels.Selector
func PodSelectorAsSelector(ps *PodSelector) (labels.Selector, error) {
	if ps == nil {
		return labels.Nothing(), nil
	}
	if len(ps.MatchLabels)+len(ps.MatchExpressions) == 0 {
		return labels.Everything(), nil
	}
	selector := labels.LabelSelector{}
	for k, v := range ps.MatchLabels {
		req, err := labels.NewRequirement(k, labels.InOperator, sets.NewString(v))
		if err != nil {
			return nil, err
		}
		selector = append(selector, *req)
	}
	for _, expr := range ps.MatchExpressions {
		var op labels.Operator
		switch expr.Operator {
		case PodSelectorOpIn:
			op = labels.InOperator
		case PodSelectorOpNotIn:
			op = labels.NotInOperator
		case PodSelectorOpExists:
			op = labels.ExistsOperator
		case PodSelectorOpDoesNotExist:
			op = labels.DoesNotExistOperator
		default:
			return nil, fmt.Errorf("%q is not a valid pod selector operator", expr.Operator)
		}
		req, err := labels.NewRequirement(expr.Key, op, sets.NewString(expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = append(selector, *req)
	}
	sort.Sort(labels.ByKey(selector))
	return selector, nil
}
