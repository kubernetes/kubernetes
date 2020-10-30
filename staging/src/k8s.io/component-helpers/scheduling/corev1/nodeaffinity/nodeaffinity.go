/*
Copyright 2020 The Kubernetes Authors.

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

package nodeaffinity

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/errors"
)

// NodeSelector is a runtime representation of v1.NodeSelector.
type NodeSelector struct {
	lazy LazyErrorNodeSelector
}

// LazyErrorNodeSelector is a runtime representation of v1.NodeSelector that
// only reports parse errors when no terms match.
type LazyErrorNodeSelector struct {
	terms []nodeSelectorTerm
}

// NewNodeSelector returns a NodeSelector or all parsing errors found.
func NewNodeSelector(ns *v1.NodeSelector) (*NodeSelector, error) {
	lazy := NewLazyErrorNodeSelector(ns)
	var errs []error
	for _, term := range lazy.terms {
		if term.parseErr != nil {
			errs = append(errs, term.parseErr)
		}
	}
	if len(errs) != 0 {
		return nil, errors.NewAggregate(errs)
	}
	return &NodeSelector{lazy: *lazy}, nil
}

// NewLazyErrorNodeSelector creates a NodeSelector that only reports parse
// errors when no terms match.
func NewLazyErrorNodeSelector(ns *v1.NodeSelector) *LazyErrorNodeSelector {
	parsedTerms := make([]nodeSelectorTerm, 0, len(ns.NodeSelectorTerms))
	for _, term := range ns.NodeSelectorTerms {
		// nil or empty term selects no objects
		if len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0 {
			continue
		}
		parsedTerms = append(parsedTerms, nodeSelectorTerm{})
		parsedTerm := &parsedTerms[len(parsedTerms)-1]
		if len(term.MatchExpressions) != 0 {
			parsedTerm.matchLabels, parsedTerm.parseErr = nodeSelectorRequirementsAsSelector(term.MatchExpressions)
			if parsedTerm.parseErr != nil {
				continue
			}
		}
		if len(term.MatchFields) != 0 {
			parsedTerm.matchFields, parsedTerm.parseErr = nodeSelectorRequirementsAsFieldSelector(term.MatchFields)
		}
	}
	return &LazyErrorNodeSelector{
		terms: parsedTerms,
	}
}

// Match checks whether the node labels and fields match the selector terms, ORed;
// nil or empty term matches no objects.
func (ns *NodeSelector) Match(node *v1.Node) bool {
	// parse errors are reported in NewNodeSelector.
	match, _ := ns.lazy.Match(node)
	return match
}

// Match checks whether the node labels and fields match the selector terms, ORed;
// nil or empty term matches no objects.
// Parse errors are only returned if no terms matched.
func (ns *LazyErrorNodeSelector) Match(node *v1.Node) (bool, error) {
	if node == nil {
		return false, nil
	}
	nodeLabels := labels.Set(node.Labels)
	nodeFields := make(fields.Set)
	if len(node.Name) > 0 {
		nodeFields["metadata.name"] = node.Name
	}

	var errs []error
	for _, term := range ns.terms {
		match, err := term.match(nodeLabels, nodeFields)
		if err != nil {
			errs = append(errs, term.parseErr)
			continue
		}
		if match {
			return true, nil
		}
	}
	return false, errors.NewAggregate(errs)
}

type nodeSelectorTerm struct {
	matchLabels labels.Selector
	matchFields fields.Selector
	parseErr    error
}

func (t *nodeSelectorTerm) match(nodeLabels labels.Set, nodeFields fields.Set) (bool, error) {
	if t.parseErr != nil {
		return false, t.parseErr
	}
	if t.matchLabels != nil && !t.matchLabels.Matches(nodeLabels) {
		return false, nil
	}
	if t.matchFields != nil && len(nodeFields) > 0 && !t.matchFields.Matches(nodeFields) {
		return false, nil
	}
	return true, nil
}

// nodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func nodeSelectorRequirementsAsSelector(nsm []v1.NodeSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op selection.Operator
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			op = selection.In
		case v1.NodeSelectorOpNotIn:
			op = selection.NotIn
		case v1.NodeSelectorOpExists:
			op = selection.Exists
		case v1.NodeSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		case v1.NodeSelectorOpGt:
			op = selection.GreaterThan
		case v1.NodeSelectorOpLt:
			op = selection.LessThan
		default:
			return nil, fmt.Errorf("%q is not a valid node selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, expr.Values)
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}

// nodeSelectorRequirementsAsFieldSelector converts the []NodeSelectorRequirement core type into a struct that implements
// fields.Selector.
func nodeSelectorRequirementsAsFieldSelector(nsr []v1.NodeSelectorRequirement) (fields.Selector, error) {
	if len(nsr) == 0 {
		return fields.Nothing(), nil
	}

	var selectors []fields.Selector
	for _, expr := range nsr {
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			if len(expr.Values) != 1 {
				return nil, fmt.Errorf("unexpected number of value (%d) for node field selector operator %q",
					len(expr.Values), expr.Operator)
			}
			selectors = append(selectors, fields.OneTermEqualSelector(expr.Key, expr.Values[0]))

		case v1.NodeSelectorOpNotIn:
			if len(expr.Values) != 1 {
				return nil, fmt.Errorf("unexpected number of value (%d) for node field selector operator %q",
					len(expr.Values), expr.Operator)
			}
			selectors = append(selectors, fields.OneTermNotEqualSelector(expr.Key, expr.Values[0]))

		default:
			return nil, fmt.Errorf("%q is not a valid node field selector operator", expr.Operator)
		}
	}

	return fields.AndSelectors(selectors...), nil
}
