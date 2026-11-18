/*
Copyright The Kubernetes Authors.

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

package api

import (
	"encoding/json"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NodeSelector is a v1.NodeSelector which was compiled once while converting
// a ResourceSlice, instead of on each match like
// k8s.io/component-helpers/scheduling/corev1/nodeaffinity does. Strings are
// unique. Matching has the same semantic as nodeaffinity.NodeSelector.
//
// +k8s:conversion-gen=false
// +k8s:deepcopy-gen=false
type NodeSelector struct {
	// Source is the original selector. It is needed when constructing
	// an allocation result and for converting back to the real API.
	Source *v1.NodeSelector

	terms []nodeSelectorTerm
	// err is the error from compiling Source. It gets reported by Match
	// because conversion must not fail because of it.
	err error
}

type nodeSelectorTerm struct {
	labels []labelRequirement
	fields []fieldRequirement
}

type labelRequirement struct {
	key UniqueString
	op  v1.NodeSelectorOperator
	// values is used for In and NotIn.
	values []UniqueString
	// bound is used for Gt and Lt.
	bound int64
}

type fieldRequirement struct {
	// Only metadata.name is known. Other fields always have an empty value.
	isName bool
	value  UniqueString
	equal  bool
}

// DeepCopyInto copies the receiver into out. The compiled terms are
// immutable and thus shared.
func (ns *NodeSelector) DeepCopyInto(out *NodeSelector) {
	*out = *ns
	out.Source = ns.Source.DeepCopy()
}

func (ns *NodeSelector) DeepCopy() *NodeSelector {
	if ns == nil {
		return nil
	}
	out := new(NodeSelector)
	ns.DeepCopyInto(out)
	return out
}

// Original returns the selector from which this one was compiled,
// or nil if the receiver is nil.
func (ns *NodeSelector) Original() *v1.NodeSelector {
	if ns == nil {
		return nil
	}
	return ns.Source
}

// String returns the same text as the original selector.
func (ns *NodeSelector) String() string {
	if ns == nil {
		return "nil"
	}
	return ns.Source.String()
}

// MarshalJSON is primarily useful for pretty-printing as JSON or YAML.
func (ns *NodeSelector) MarshalJSON() ([]byte, error) {
	return json.Marshal(ns.Source)
}

// newNodeSelector compiles the selector. Empty terms select nothing and
// are dropped. Any invalid term is reported by Match.
func newNodeSelector(source *v1.NodeSelector, makeUniqueString func(string) UniqueString) *NodeSelector {
	ns := &NodeSelector{Source: source}
	var errs []error
	path := field.NewPath("nodeSelectorTerms")
	for i, term := range source.NodeSelectorTerms {
		if len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0 {
			continue
		}
		compiled, termErrs := compileNodeSelectorTerm(&term, path.Index(i), makeUniqueString)
		errs = append(errs, termErrs...)
		ns.terms = append(ns.terms, compiled)
	}
	if len(errs) > 0 {
		ns.err = utilerrors.Flatten(utilerrors.NewAggregate(errs))
		ns.terms = nil
	}
	return ns
}

func compileNodeSelectorTerm(term *v1.NodeSelectorTerm, path *field.Path, makeUniqueString func(string) UniqueString) (nodeSelectorTerm, []error) {
	var result nodeSelectorTerm
	var errs []error

	p := path.Child("matchExpressions")
	for i, expr := range term.MatchExpressions {
		p := p.Index(i)
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
			errs = append(errs, field.NotSupported(p.Child("operator"), expr.Operator, []v1.NodeSelectorOperator{
				v1.NodeSelectorOpIn, v1.NodeSelectorOpNotIn, v1.NodeSelectorOpExists,
				v1.NodeSelectorOpDoesNotExist, v1.NodeSelectorOpGt, v1.NodeSelectorOpLt,
			}))
			continue
		}
		// Validates key and values.
		if _, err := labels.NewRequirement(expr.Key, op, expr.Values, field.WithPath(p)); err != nil {
			errs = append(errs, err)
			continue
		}
		req := labelRequirement{key: makeUniqueString(expr.Key), op: expr.Operator}
		switch expr.Operator {
		case v1.NodeSelectorOpIn, v1.NodeSelectorOpNotIn:
			req.values = make([]UniqueString, len(expr.Values))
			for i, value := range expr.Values {
				req.values[i] = makeUniqueString(value)
			}
		case v1.NodeSelectorOpGt, v1.NodeSelectorOpLt:
			// Validated above.
			req.bound, _ = strconv.ParseInt(expr.Values[0], 10, 64)
		}
		result.labels = append(result.labels, req)
	}

	p = path.Child("matchFields")
	for i, expr := range term.MatchFields {
		p := p.Index(i)
		switch expr.Operator {
		case v1.NodeSelectorOpIn, v1.NodeSelectorOpNotIn:
			if len(expr.Values) != 1 {
				errs = append(errs, field.Invalid(p.Child("values"), expr.Values, "must have one element"))
				continue
			}
			result.fields = append(result.fields, fieldRequirement{
				isName: expr.Key == "metadata.name",
				value:  makeUniqueString(expr.Values[0]),
				equal:  expr.Operator == v1.NodeSelectorOpIn,
			})
		default:
			errs = append(errs, field.NotSupported(p.Child("operator"), expr.Operator, []v1.NodeSelectorOperator{
				v1.NodeSelectorOpIn, v1.NodeSelectorOpNotIn,
			}))
		}
	}

	return result, errs
}

// Match checks whether the node labels and fields match the selector terms, ORed.
// It returns an error if the selector is invalid.
func (ns *NodeSelector) Match(node *v1.Node) (bool, error) {
	if ns.err != nil {
		return false, ns.err
	}
	if node == nil {
		return false, nil
	}
	for i := range ns.terms {
		if ns.terms[i].match(node) {
			return true, nil
		}
	}
	return false, nil
}

func (t *nodeSelectorTerm) match(node *v1.Node) bool {
	for i := range t.labels {
		if !t.labels[i].matches(node.Labels) {
			return false
		}
	}
	// As in nodeaffinity, field selection is skipped for nodes without name.
	if node.Name == "" {
		return true
	}
	for _, req := range t.fields {
		value := ""
		if req.isName {
			value = node.Name
		}
		if (value == req.value.String()) != req.equal {
			return false
		}
	}
	return true
}

func (r *labelRequirement) matches(nodeLabels map[string]string) bool {
	value, found := nodeLabels[r.key.String()]
	switch r.op {
	case v1.NodeSelectorOpIn, v1.NodeSelectorOpNotIn:
		if !found {
			return r.op == v1.NodeSelectorOpNotIn
		}
		for _, v := range r.values {
			if v.String() == value {
				return r.op == v1.NodeSelectorOpIn
			}
		}
		return r.op == v1.NodeSelectorOpNotIn
	case v1.NodeSelectorOpExists:
		return found
	case v1.NodeSelectorOpDoesNotExist:
		return !found
	case v1.NodeSelectorOpGt, v1.NodeSelectorOpLt:
		if !found {
			return false
		}
		n, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return false
		}
		if r.op == v1.NodeSelectorOpGt {
			return n > r.bound
		}
		return n < r.bound
	}
	return false
}
