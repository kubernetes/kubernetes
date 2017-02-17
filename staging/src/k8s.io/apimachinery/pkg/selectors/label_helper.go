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

package selectors

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
)

// LabelSelectorAsSelector converts the LabelSelector api type into a struct that implements
// labels.Selector
// Note: This function should be kept in sync with the selector methods in pkg/labels/selector.go
func LabelSelectorAsSelector(ps *metav1.LabelSelector) (labels.Selector, error) {
	if ps == nil {
		return labels.Nothing(), nil
	}
	if len(ps.MatchLabels)+len(ps.MatchExpressions) == 0 {
		return labels.Everything(), nil
	}
	selector := labels.NewSelector()
	for k, v := range ps.MatchLabels {
		r, err := labels.NewRequirement(k, selection.Equals, []string{v})
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	for _, expr := range ps.MatchExpressions {
		var op selection.Operator
		switch expr.Operator {
		case metav1.LabelSelectorOpIn:
			op = selection.In
		case metav1.LabelSelectorOpNotIn:
			op = selection.NotIn
		case metav1.LabelSelectorOpExists:
			op = selection.Exists
		case metav1.LabelSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		default:
			return nil, fmt.Errorf("%q is not a valid pod selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, append([]string(nil), expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}

// LabelSelectorAsMap converts the LabelSelector api type into a map of strings, ie. the
// original structure of a label selector. Operators that cannot be converted into plain
// labels (Exists, DoesNotExist, NotIn, and In with more than one value) will result in
// an error.
func LabelSelectorAsMap(ps *metav1.LabelSelector) (map[string]string, error) {
	if ps == nil {
		return nil, nil
	}
	selector := map[string]string{}
	for k, v := range ps.MatchLabels {
		selector[k] = v
	}
	for _, expr := range ps.MatchExpressions {
		switch expr.Operator {
		case metav1.LabelSelectorOpIn:
			if len(expr.Values) != 1 {
				return selector, fmt.Errorf("operator %q without a single value cannot be converted into the old label selector format", expr.Operator)
			}
			// Should we do anything in case this will override a previous key-value pair?
			selector[expr.Key] = expr.Values[0]
		case metav1.LabelSelectorOpNotIn, metav1.LabelSelectorOpExists, metav1.LabelSelectorOpDoesNotExist:
			return selector, fmt.Errorf("operator %q cannot be converted into the old label selector format", expr.Operator)
		default:
			return selector, fmt.Errorf("%q is not a valid selector operator", expr.Operator)
		}
	}
	return selector, nil
}

// ParseToLabelSelector parses a string representing a selector into a LabelSelector object.
// Note: This function should be kept in sync with the parser in pkg/labels/selector.go
func ParseToLabelSelector(selector string) (*metav1.LabelSelector, error) {
	reqs, err := labels.ParseToRequirements(selector)
	if err != nil {
		return nil, fmt.Errorf("couldn't parse the selector string \"%s\": %v", selector, err)
	}

	labelSelector := &metav1.LabelSelector{
		MatchLabels:      map[string]string{},
		MatchExpressions: []metav1.LabelSelectorRequirement{},
	}
	for _, req := range reqs {
		var op metav1.LabelSelectorOperator
		switch req.Operator() {
		case selection.Equals, selection.DoubleEquals:
			vals := req.Values()
			if vals.Len() != 1 {
				return nil, fmt.Errorf("equals operator must have exactly one value")
			}
			val, ok := vals.PopAny()
			if !ok {
				return nil, fmt.Errorf("equals operator has exactly one value but it cannot be retrieved")
			}
			labelSelector.MatchLabels[req.Key()] = val
			continue
		case selection.In:
			op = metav1.LabelSelectorOpIn
		case selection.NotIn:
			op = metav1.LabelSelectorOpNotIn
		case selection.Exists:
			op = metav1.LabelSelectorOpExists
		case selection.DoesNotExist:
			op = metav1.LabelSelectorOpDoesNotExist
		case selection.GreaterThan, selection.LessThan:
			// Adding a separate case for these operators to indicate that this is deliberate
			return nil, fmt.Errorf("%q isn't supported in label selectors", req.Operator())
		default:
			return nil, fmt.Errorf("%q is not a valid label selector operator", req.Operator())
		}
		labelSelector.MatchExpressions = append(labelSelector.MatchExpressions, metav1.LabelSelectorRequirement{
			Key:      req.Key(),
			Operator: op,
			Values:   req.Values().List(),
		})
	}
	return labelSelector, nil
}

// SetAsLabelSelector converts the labels.Set object into a LabelSelector api object.
func SetAsLabelSelector(ls labels.Set) *metav1.LabelSelector {
	if ls == nil {
		return nil
	}

	selector := &metav1.LabelSelector{
		MatchLabels: make(map[string]string),
	}
	for label, value := range ls {
		selector.MatchLabels[label] = value
	}

	return selector
}

// FormatLabelSelector convert labelSelector into plain string
func FormatLabelSelector(labelSelector *metav1.LabelSelector) string {
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
