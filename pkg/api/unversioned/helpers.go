/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"fmt"

	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

// LabelSelectorAsSelector converts the LabelSelector api type into a struct that implements
// labels.Selector
func LabelSelectorAsSelector(ps *LabelSelector) (labels.Selector, error) {
	if ps == nil {
		return labels.Nothing(), nil
	}
	if len(ps.MatchLabels)+len(ps.MatchExpressions) == 0 {
		return labels.Everything(), nil
	}
	selector := labels.NewSelector()
	for k, v := range ps.MatchLabels {
		r, err := labels.NewRequirement(k, labels.InOperator, sets.NewString(v))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	for _, expr := range ps.MatchExpressions {
		var op labels.Operator
		switch expr.Operator {
		case LabelSelectorOpIn:
			op = labels.InOperator
		case LabelSelectorOpNotIn:
			op = labels.NotInOperator
		case LabelSelectorOpExists:
			op = labels.ExistsOperator
		case LabelSelectorOpDoesNotExist:
			op = labels.DoesNotExistOperator
		default:
			return nil, fmt.Errorf("%q is not a valid pod selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, sets.NewString(expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}

// SetAsLabelSelector converts the labels.Set object into a LabelSelector api object.
func SetAsLabelSelector(ls labels.Set) *LabelSelector {
	if ls == nil {
		return nil
	}

	selector := &LabelSelector{
		MatchLabels: make(map[string]string),
	}
	for label, value := range ls {
		selector.MatchLabels[label] = value
	}

	return selector
}

// FormatLabelSelector convert labelSelector into plain string
func FormatLabelSelector(labelSelector *LabelSelector) string {
	selector, err := LabelSelectorAsSelector(labelSelector)
	if err != nil {
		return "<error>"
	}

	l := selector.String()
	if len(l) == 0 {
		l = "<none>"
	}
	return l
}
