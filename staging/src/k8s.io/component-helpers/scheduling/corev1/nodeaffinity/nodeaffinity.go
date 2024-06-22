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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
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

// NewNodeSelector returns a NodeSelector or aggregate parsing errors found.
func NewNodeSelector(ns *v1.NodeSelector, opts ...field.PathOption) (*NodeSelector, error) {
	lazy := NewLazyErrorNodeSelector(ns, opts...)
	var errs []error
	for _, term := range lazy.terms {
		if len(term.parseErrs) > 0 {
			errs = append(errs, term.parseErrs...)
		}
	}
	if len(errs) != 0 {
		return nil, errors.Flatten(errors.NewAggregate(errs))
	}
	return &NodeSelector{lazy: *lazy}, nil
}

// NewLazyErrorNodeSelector creates a NodeSelector that only reports parse
// errors when no terms match.
func NewLazyErrorNodeSelector(ns *v1.NodeSelector, opts ...field.PathOption) *LazyErrorNodeSelector {
	p := field.ToPath(opts...)
	parsedTerms := make([]nodeSelectorTerm, 0, len(ns.NodeSelectorTerms))
	path := p.Child("nodeSelectorTerms")
	for i, term := range ns.NodeSelectorTerms {
		// nil or empty term selects no objects
		if isEmptyNodeSelectorTerm(&term) {
			continue
		}
		p := path.Index(i)
		parsedTerms = append(parsedTerms, newNodeSelectorTerm(&term, p))
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
	nodeFields := extractNodeFields(node)

	var errs []error
	for _, term := range ns.terms {
		match, tErrs := term.match(nodeLabels, nodeFields)
		if len(tErrs) > 0 {
			errs = append(errs, tErrs...)
			continue
		}
		if match {
			return true, nil
		}
	}
	return false, errors.Flatten(errors.NewAggregate(errs))
}

// PreferredSchedulingTerms is a runtime representation of []v1.PreferredSchedulingTerms.
type PreferredSchedulingTerms struct {
	terms []preferredSchedulingTerm
}

// NewPreferredSchedulingTerms returns a PreferredSchedulingTerms or all the parsing errors found.
// If a v1.PreferredSchedulingTerm has a 0 weight, its parsing is skipped.
func NewPreferredSchedulingTerms(terms []v1.PreferredSchedulingTerm, opts ...field.PathOption) (*PreferredSchedulingTerms, error) {
	p := field.ToPath(opts...)
	var errs []error
	parsedTerms := make([]preferredSchedulingTerm, 0, len(terms))
	for i, term := range terms {
		path := p.Index(i)
		if term.Weight == 0 || isEmptyNodeSelectorTerm(&term.Preference) {
			continue
		}
		parsedTerm := preferredSchedulingTerm{
			nodeSelectorTerm: newNodeSelectorTerm(&term.Preference, path),
			weight:           int(term.Weight),
		}
		if len(parsedTerm.parseErrs) > 0 {
			errs = append(errs, parsedTerm.parseErrs...)
		} else {
			parsedTerms = append(parsedTerms, parsedTerm)
		}
	}
	if len(errs) != 0 {
		return nil, errors.Flatten(errors.NewAggregate(errs))
	}
	return &PreferredSchedulingTerms{terms: parsedTerms}, nil
}

// Score returns a score for a Node: the sum of the weights of the terms that
// match the Node.
func (t *PreferredSchedulingTerms) Score(node *v1.Node) int64 {
	var score int64
	nodeLabels := labels.Set(node.Labels)
	nodeFields := extractNodeFields(node)
	for _, term := range t.terms {
		// parse errors are reported in NewPreferredSchedulingTerms.
		if ok, _ := term.match(nodeLabels, nodeFields); ok {
			score += int64(term.weight)
		}
	}
	return score
}

func isEmptyNodeSelectorTerm(term *v1.NodeSelectorTerm) bool {
	return len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0
}

func extractNodeFields(n *v1.Node) fields.Set {
	f := make(fields.Set)
	if len(n.Name) > 0 {
		f["metadata.name"] = n.Name
	}
	return f
}

type nodeSelectorTerm struct {
	matchLabels labels.Selector
	matchFields fields.Selector
	parseErrs   []error
}

func newNodeSelectorTerm(term *v1.NodeSelectorTerm, path *field.Path) nodeSelectorTerm {
	var parsedTerm nodeSelectorTerm
	var errs []error
	if len(term.MatchExpressions) != 0 {
		p := path.Child("matchExpressions")
		parsedTerm.matchLabels, errs = nodeSelectorRequirementsAsSelector(term.MatchExpressions, p)
		if errs != nil {
			parsedTerm.parseErrs = append(parsedTerm.parseErrs, errs...)
		}
	}
	if len(term.MatchFields) != 0 {
		p := path.Child("matchFields")
		parsedTerm.matchFields, errs = nodeSelectorRequirementsAsFieldSelector(term.MatchFields, p)
		if errs != nil {
			parsedTerm.parseErrs = append(parsedTerm.parseErrs, errs...)
		}
	}
	return parsedTerm
}

func (t *nodeSelectorTerm) match(nodeLabels labels.Set, nodeFields fields.Set) (bool, []error) {
	if t.parseErrs != nil {
		return false, t.parseErrs
	}
	if t.matchLabels != nil && !t.matchLabels.Matches(nodeLabels) {
		return false, nil
	}
	if t.matchFields != nil && len(nodeFields) > 0 && !t.matchFields.Matches(nodeFields) {
		return false, nil
	}
	return true, nil
}

var validSelectorOperators = []v1.NodeSelectorOperator{
	v1.NodeSelectorOpIn,
	v1.NodeSelectorOpNotIn,
	v1.NodeSelectorOpExists,
	v1.NodeSelectorOpDoesNotExist,
	v1.NodeSelectorOpGt,
	v1.NodeSelectorOpLt,
}

// nodeSelectorRequirementsAsSelector converts the []NodeSelectorRequirement api type into a struct that implements
// labels.Selector.
func nodeSelectorRequirementsAsSelector(nsm []v1.NodeSelectorRequirement, path *field.Path) (labels.Selector, []error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	var errs []error
	selector := labels.NewSelector()
	for i, expr := range nsm {
		p := path.Index(i)
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
			errs = append(errs, field.NotSupported(p.Child("operator"), expr.Operator, validSelectorOperators))
			continue
		}
		r, err := labels.NewRequirement(expr.Key, op, expr.Values, field.WithPath(p))
		if err != nil {
			errs = append(errs, err)
		} else {
			selector = selector.Add(*r)
		}
	}
	if len(errs) != 0 {
		return nil, errs
	}
	return selector, nil
}

var validFieldSelectorOperators = []v1.NodeSelectorOperator{
	v1.NodeSelectorOpIn,
	v1.NodeSelectorOpNotIn,
}

// nodeSelectorRequirementsAsFieldSelector converts the []NodeSelectorRequirement core type into a struct that implements
// fields.Selector.
func nodeSelectorRequirementsAsFieldSelector(nsr []v1.NodeSelectorRequirement, path *field.Path) (fields.Selector, []error) {
	if len(nsr) == 0 {
		return fields.Nothing(), nil
	}
	var errs []error

	var selectors []fields.Selector
	for i, expr := range nsr {
		p := path.Index(i)
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			if len(expr.Values) != 1 {
				errs = append(errs, field.Invalid(p.Child("values"), expr.Values, "must have one element"))
			} else {
				selectors = append(selectors, fields.OneTermEqualSelector(expr.Key, expr.Values[0]))
			}

		case v1.NodeSelectorOpNotIn:
			if len(expr.Values) != 1 {
				errs = append(errs, field.Invalid(p.Child("values"), expr.Values, "must have one element"))
			} else {
				selectors = append(selectors, fields.OneTermNotEqualSelector(expr.Key, expr.Values[0]))
			}

		default:
			errs = append(errs, field.NotSupported(p.Child("operator"), expr.Operator, validFieldSelectorOperators))
		}
	}

	if len(errs) != 0 {
		return nil, errs
	}
	return fields.AndSelectors(selectors...), nil
}

type preferredSchedulingTerm struct {
	nodeSelectorTerm
	weight int
}

type RequiredNodeAffinity struct {
	labelSelector labels.Selector
	nodeSelector  *LazyErrorNodeSelector
}

// GetRequiredNodeAffinity returns the parsing result of pod's nodeSelector and nodeAffinity.
func GetRequiredNodeAffinity(pod *v1.Pod) RequiredNodeAffinity {
	var selector labels.Selector
	if len(pod.Spec.NodeSelector) > 0 {
		selector = labels.SelectorFromSet(pod.Spec.NodeSelector)
	}
	// Use LazyErrorNodeSelector for backwards compatibility of parsing errors.
	var affinity *LazyErrorNodeSelector
	if pod.Spec.Affinity != nil &&
		pod.Spec.Affinity.NodeAffinity != nil &&
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		affinity = NewLazyErrorNodeSelector(pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution)
	}
	return RequiredNodeAffinity{labelSelector: selector, nodeSelector: affinity}
}

// Match checks whether the pod is schedulable onto nodes according to
// the requirements in both nodeSelector and nodeAffinity.
func (s RequiredNodeAffinity) Match(node *v1.Node) (bool, error) {
	if s.labelSelector != nil {
		if !s.labelSelector.Matches(labels.Set(node.Labels)) {
			return false, nil
		}
	}
	if s.nodeSelector != nil {
		return s.nodeSelector.Match(node)
	}
	return true, nil
}
