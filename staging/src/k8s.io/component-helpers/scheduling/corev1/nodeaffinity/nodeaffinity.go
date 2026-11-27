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
	"slices"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/validate/content"
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
	terms          []nodeSelectorTerm
	hasMatchFields bool
}

// MatchNodeSelectorTerms checks whether the node labels and fields match the
// selector terms in ORed fashion without allocating selector structures on valid inputs.
// Parse errors are only returned if no valid term matches.
func MatchNodeSelectorTerms(node *v1.Node, ns *v1.NodeSelector) (bool, error) {
	if node == nil {
		return false, nil
	}
	return matchNodeSelector(node.Labels, node.Name, ns)
}

// MatchNodeLabels checks whether nodeLabels match the selector terms in ORed fashion
// without allocating a temporary *v1.Node or selector structures on valid inputs.
// Parse errors (including invalid matchFields) are returned if no valid term matches.
func MatchNodeLabels(nodeLabels map[string]string, ns *v1.NodeSelector) (bool, error) {
	return matchNodeSelector(nodeLabels, "", ns)
}

func matchNodeSelector(nodeLabels map[string]string, nodeName string, ns *v1.NodeSelector) (bool, error) {
	hasInvalidTerm := false
	for i := range ns.NodeSelectorTerms {
		term := &ns.NodeSelectorTerms[i]
		if isEmptyNodeSelectorTerm(term) {
			continue
		}
		matched, valid := matchNodeSelectorTerm(term, nodeLabels, nodeName)
		if !valid {
			hasInvalidTerm = true
			continue
		}
		if matched {
			return true, nil
		}
	}
	if !hasInvalidTerm {
		return false, nil
	}
	// Slow path: at least one term had parse errors and no valid term matched.
	// Re-parse with NewLazyErrorNodeSelector to return identical aggregated field.Errors.
	lazy := NewLazyErrorNodeSelector(ns)
	var errs []error
	for i := range lazy.terms {
		if len(lazy.terms[i].parseErrs) > 0 {
			errs = append(errs, lazy.terms[i].parseErrs...)
		}
	}
	return false, errors.Flatten(errors.NewAggregate(errs))
}

func matchNodeSelectorTerm(term *v1.NodeSelectorTerm, nodeLabels map[string]string, nodeName string) (matched bool, valid bool) {
	matched = true
	for i := range term.MatchExpressions {
		expr := &term.MatchExpressions[i]
		if len(content.IsLabelKey(expr.Key)) != 0 {
			return false, false
		}
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			if len(expr.Values) == 0 {
				return false, false
			}
			for _, v := range expr.Values {
				if len(content.IsLabelValue(v)) != 0 {
					return false, false
				}
			}
			if matched {
				val, exists := nodeLabels[expr.Key]
				if !exists || !slices.Contains(expr.Values, val) {
					matched = false
				}
			}
		case v1.NodeSelectorOpNotIn:
			if len(expr.Values) == 0 {
				return false, false
			}
			for _, v := range expr.Values {
				if len(content.IsLabelValue(v)) != 0 {
					return false, false
				}
			}
			if matched {
				val, exists := nodeLabels[expr.Key]
				if exists && slices.Contains(expr.Values, val) {
					matched = false
				}
			}
		case v1.NodeSelectorOpExists:
			if len(expr.Values) != 0 {
				return false, false
			}
			if matched {
				if _, exists := nodeLabels[expr.Key]; !exists {
					matched = false
				}
			}
		case v1.NodeSelectorOpDoesNotExist:
			if len(expr.Values) != 0 {
				return false, false
			}
			if matched {
				if _, exists := nodeLabels[expr.Key]; exists {
					matched = false
				}
			}
		case v1.NodeSelectorOpGt, v1.NodeSelectorOpLt:
			if len(expr.Values) != 1 {
				return false, false
			}
			if len(content.IsLabelValue(expr.Values[0])) != 0 {
				return false, false
			}
			rVal, err := strconv.ParseInt(expr.Values[0], 10, 64)
			if err != nil {
				return false, false
			}
			if matched {
				val, exists := nodeLabels[expr.Key]
				if !exists {
					matched = false
				} else if lsVal, err := strconv.ParseInt(val, 10, 64); err != nil {
					matched = false
				} else if expr.Operator == v1.NodeSelectorOpGt {
					if lsVal <= rVal {
						matched = false
					}
				} else {
					if lsVal >= rVal {
						matched = false
					}
				}
			}
		default:
			return false, false
		}
	}

	for i := range term.MatchFields {
		expr := &term.MatchFields[i]
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			if len(expr.Values) != 1 {
				return false, false
			}
			if matched && len(nodeName) > 0 {
				var fieldVal string
				if expr.Key == "metadata.name" {
					fieldVal = nodeName
				}
				if fieldVal != expr.Values[0] {
					matched = false
				}
			}
		case v1.NodeSelectorOpNotIn:
			if len(expr.Values) != 1 {
				return false, false
			}
			if matched && len(nodeName) > 0 {
				var fieldVal string
				if expr.Key == "metadata.name" {
					fieldVal = nodeName
				}
				if fieldVal == expr.Values[0] {
					matched = false
				}
			}
		default:
			return false, false
		}
	}

	return matched, true
}

type lazyTermPath struct {
	basePath    *field.Path
	useBasePath bool
	opts        []field.PathOption
	termsChild  string
	termIdx     int
}

func (lp lazyTermPath) exprPath(child string, idx int) *field.Path {
	if lp.useBasePath {
		return lp.basePath.Index(idx)
	}
	p := field.ToPath(lp.opts...)
	if lp.termsChild != "" {
		p = p.Child(lp.termsChild)
	}
	return p.Index(lp.termIdx).Child(child).Index(idx)
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
	parsedTerms := make([]nodeSelectorTerm, 0, len(ns.NodeSelectorTerms))
	hasMatchFields := false
	for i := range ns.NodeSelectorTerms {
		term := &ns.NodeSelectorTerms[i]
		// nil or empty term selects no objects
		if isEmptyNodeSelectorTerm(term) {
			continue
		}
		lp := lazyTermPath{opts: opts, termsChild: "nodeSelectorTerms", termIdx: i}
		parsedTerm := newNodeSelectorTerm(term, lp)
		if parsedTerm.matchFields != nil {
			hasMatchFields = true
		}
		parsedTerms = append(parsedTerms, parsedTerm)
	}
	return &LazyErrorNodeSelector{
		terms:          parsedTerms,
		hasMatchFields: hasMatchFields,
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
	if node == nil || len(ns.terms) == 0 {
		return false, nil
	}
	nodeLabels := labels.Set(node.Labels)
	var nodeFields fields.Set
	if ns.hasMatchFields && len(node.Name) > 0 {
		nodeFields = extractNodeFields(node)
	}

	var errs []error
	for i := range ns.terms {
		match, tErrs := ns.terms[i].match(nodeLabels, nodeFields)
		if len(tErrs) > 0 {
			errs = append(errs, tErrs...)
			continue
		}
		if match {
			return true, nil
		}
	}
	if len(errs) == 0 {
		return false, nil
	}
	return false, errors.Flatten(errors.NewAggregate(errs))
}

// MatchLabels checks whether the node labels match the selector terms, ORed;
// nil or empty term matches no objects.
// Parse errors are only returned if no terms matched.
// MatchFields are not evaluated (equivalent to Match with an empty node name),
// while any MatchFields parse errors are still preserved.
func (ns *LazyErrorNodeSelector) MatchLabels(nodeLabels labels.Set) (bool, error) {
	if len(ns.terms) == 0 {
		return false, nil
	}
	var errs []error
	for i := range ns.terms {
		match, tErrs := ns.terms[i].match(nodeLabels, nil)
		if len(tErrs) > 0 {
			errs = append(errs, tErrs...)
			continue
		}
		if match {
			return true, nil
		}
	}
	if len(errs) == 0 {
		return false, nil
	}
	return false, errors.Flatten(errors.NewAggregate(errs))
}

// PreferredSchedulingTerms is a runtime representation of []v1.PreferredSchedulingTerms.
type PreferredSchedulingTerms struct {
	terms          []preferredSchedulingTerm
	hasMatchFields bool
}

// NewPreferredSchedulingTerms returns a PreferredSchedulingTerms or all the parsing errors found.
// If a v1.PreferredSchedulingTerm has a 0 weight, its parsing is skipped.
func NewPreferredSchedulingTerms(terms []v1.PreferredSchedulingTerm, opts ...field.PathOption) (*PreferredSchedulingTerms, error) {
	var errs []error
	parsedTerms := make([]preferredSchedulingTerm, 0, len(terms))
	hasMatchFields := false
	for i := range terms {
		term := &terms[i]
		if term.Weight == 0 || isEmptyNodeSelectorTerm(&term.Preference) {
			continue
		}
		lp := lazyTermPath{opts: opts, termIdx: i}
		parsedTerm := preferredSchedulingTerm{
			nodeSelectorTerm: newNodeSelectorTerm(&term.Preference, lp),
			weight:           int(term.Weight),
		}
		if len(parsedTerm.parseErrs) > 0 {
			errs = append(errs, parsedTerm.parseErrs...)
		} else {
			if parsedTerm.matchFields != nil {
				hasMatchFields = true
			}
			parsedTerms = append(parsedTerms, parsedTerm)
		}
	}
	if len(errs) != 0 {
		return nil, errors.Flatten(errors.NewAggregate(errs))
	}
	return &PreferredSchedulingTerms{terms: parsedTerms, hasMatchFields: hasMatchFields}, nil
}

// Score returns a score for a Node: the sum of the weights of the terms that
// match the Node.
func (t *PreferredSchedulingTerms) Score(node *v1.Node) int64 {
	var score int64
	nodeLabels := labels.Set(node.Labels)
	var nodeFields fields.Set
	if t.hasMatchFields && len(node.Name) > 0 {
		nodeFields = extractNodeFields(node)
	}
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

func newNodeSelectorTerm(term *v1.NodeSelectorTerm, lp lazyTermPath) nodeSelectorTerm {
	var parsedTerm nodeSelectorTerm
	var errs []error
	if len(term.MatchExpressions) != 0 {
		parsedTerm.matchLabels, errs = nodeSelectorRequirementsAsSelectorLazy(term.MatchExpressions, lp)
		if errs != nil {
			parsedTerm.parseErrs = append(parsedTerm.parseErrs, errs...)
		}
	}
	if len(term.MatchFields) != 0 {
		parsedTerm.matchFields, errs = nodeSelectorRequirementsAsFieldSelectorLazy(term.MatchFields, lp)
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
	return nodeSelectorRequirementsAsSelectorLazy(nsm, lazyTermPath{basePath: path, useBasePath: true})
}

func nodeSelectorRequirementsAsSelectorLazy(nsm []v1.NodeSelectorRequirement, lp lazyTermPath) (labels.Selector, []error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	var errs []error
	reqs := make([]labels.Requirement, 0, len(nsm))
	for i, expr := range nsm {
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
			errs = append(errs, field.NotSupported(lp.exprPath("matchExpressions", i).Child("operator"), expr.Operator, validSelectorOperators))
			continue
		}
		r, err := labels.NewRequirement(expr.Key, op, expr.Values)
		if err != nil {
			_, err = labels.NewRequirement(expr.Key, op, expr.Values, field.WithPath(lp.exprPath("matchExpressions", i)))
			errs = append(errs, err)
		} else {
			reqs = append(reqs, *r)
		}
	}
	if len(errs) != 0 {
		return nil, errs
	}
	return labels.NewSelector().Add(reqs...), nil
}

var validFieldSelectorOperators = []v1.NodeSelectorOperator{
	v1.NodeSelectorOpIn,
	v1.NodeSelectorOpNotIn,
}

// nodeSelectorRequirementsAsFieldSelector converts the []NodeSelectorRequirement core type into a struct that implements
// fields.Selector.
func nodeSelectorRequirementsAsFieldSelector(nsr []v1.NodeSelectorRequirement, path *field.Path) (fields.Selector, []error) {
	return nodeSelectorRequirementsAsFieldSelectorLazy(nsr, lazyTermPath{basePath: path, useBasePath: true})
}

func nodeSelectorRequirementsAsFieldSelectorLazy(nsr []v1.NodeSelectorRequirement, lp lazyTermPath) (fields.Selector, []error) {
	if len(nsr) == 0 {
		return fields.Nothing(), nil
	}
	var errs []error

	selectors := make([]fields.Selector, 0, len(nsr))
	for i, expr := range nsr {
		switch expr.Operator {
		case v1.NodeSelectorOpIn:
			if len(expr.Values) != 1 {
				errs = append(errs, field.Invalid(lp.exprPath("matchFields", i).Child("values"), expr.Values, "must have one element"))
			} else {
				selectors = append(selectors, fields.OneTermEqualSelector(expr.Key, expr.Values[0]))
			}

		case v1.NodeSelectorOpNotIn:
			if len(expr.Values) != 1 {
				errs = append(errs, field.Invalid(lp.exprPath("matchFields", i).Child("values"), expr.Values, "must have one element"))
			} else {
				selectors = append(selectors, fields.OneTermNotEqualSelector(expr.Key, expr.Values[0]))
			}

		default:
			errs = append(errs, field.NotSupported(lp.exprPath("matchFields", i).Child("operator"), expr.Operator, validFieldSelectorOperators))
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
	return NewRequiredNodeAffinity(pod.Spec.NodeSelector, pod.Spec.Affinity)
}

// NewRequiredNodeAffinity returns the parsing result of the given nodeSelector
// and affinity. It is equivalent to GetRequiredNodeAffinity for a pod carrying
// them, for callers that have a pod spec's fields but no pod object (e.g.
// workload controllers evaluating a pod template).
func NewRequiredNodeAffinity(nodeSelector map[string]string, affinity *v1.Affinity) RequiredNodeAffinity {
	var selector labels.Selector
	if len(nodeSelector) > 0 {
		selector = labels.SelectorFromSet(nodeSelector)
	}
	// Use LazyErrorNodeSelector for backwards compatibility of parsing errors.
	var lazyNodeSelector *LazyErrorNodeSelector
	if affinity != nil &&
		affinity.NodeAffinity != nil &&
		affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		lazyNodeSelector = NewLazyErrorNodeSelector(affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution)
	}
	return RequiredNodeAffinity{labelSelector: selector, nodeSelector: lazyNodeSelector}
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
