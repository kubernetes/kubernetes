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

package v1

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	utiljson "k8s.io/apimachinery/pkg/util/json"
)

// LabelSelectorAsSelector converts the LabelSelector api type into a struct that implements
// labels.Selector
// Note: This function should be kept in sync with the selector methods in pkg/labels/selector.go
func LabelSelectorAsSelector(ps *LabelSelector) (labels.Selector, error) {
	if ps == nil {
		return labels.Nothing(), nil
	}
	if len(ps.MatchLabels)+len(ps.MatchExpressions) == 0 {
		return labels.Everything(), nil
	}
	requirements := make([]labels.Requirement, 0, len(ps.MatchLabels)+len(ps.MatchExpressions))
	for k, v := range ps.MatchLabels {
		r, err := labels.NewRequirement(k, selection.Equals, []string{v})
		if err != nil {
			return nil, err
		}
		requirements = append(requirements, *r)
	}
	for _, expr := range ps.MatchExpressions {
		var op selection.Operator
		switch expr.Operator {
		case LabelSelectorOpIn:
			op = selection.In
		case LabelSelectorOpNotIn:
			op = selection.NotIn
		case LabelSelectorOpExists:
			op = selection.Exists
		case LabelSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		default:
			return nil, fmt.Errorf("%q is not a valid label selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, append([]string(nil), expr.Values...))
		if err != nil {
			return nil, err
		}
		requirements = append(requirements, *r)
	}
	selector := labels.NewSelector()
	selector = selector.Add(requirements...)
	return selector, nil
}

// LabelSelectorAsMap converts the LabelSelector api type into a map of strings, ie. the
// original structure of a label selector. Operators that cannot be converted into plain
// labels (Exists, DoesNotExist, NotIn, and In with more than one value) will result in
// an error.
func LabelSelectorAsMap(ps *LabelSelector) (map[string]string, error) {
	if ps == nil {
		return nil, nil
	}
	selector := map[string]string{}
	for k, v := range ps.MatchLabels {
		selector[k] = v
	}
	for _, expr := range ps.MatchExpressions {
		switch expr.Operator {
		case LabelSelectorOpIn:
			if len(expr.Values) != 1 {
				return selector, fmt.Errorf("operator %q without a single value cannot be converted into the old label selector format", expr.Operator)
			}
			// Should we do anything in case this will override a previous key-value pair?
			selector[expr.Key] = expr.Values[0]
		case LabelSelectorOpNotIn, LabelSelectorOpExists, LabelSelectorOpDoesNotExist:
			return selector, fmt.Errorf("operator %q cannot be converted into the old label selector format", expr.Operator)
		default:
			return selector, fmt.Errorf("%q is not a valid selector operator", expr.Operator)
		}
	}
	return selector, nil
}

// ParseToLabelSelector parses a string representing a selector into a LabelSelector object.
// Note: This function should be kept in sync with the parser in pkg/labels/selector.go
func ParseToLabelSelector(selector string) (*LabelSelector, error) {
	reqs, err := labels.ParseToRequirements(selector)
	if err != nil {
		return nil, fmt.Errorf("couldn't parse the selector string \"%s\": %v", selector, err)
	}

	labelSelector := &LabelSelector{
		MatchLabels:      map[string]string{},
		MatchExpressions: []LabelSelectorRequirement{},
	}
	for _, req := range reqs {
		var op LabelSelectorOperator
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
			op = LabelSelectorOpIn
		case selection.NotIn:
			op = LabelSelectorOpNotIn
		case selection.Exists:
			op = LabelSelectorOpExists
		case selection.DoesNotExist:
			op = LabelSelectorOpDoesNotExist
		case selection.GreaterThan, selection.LessThan:
			// Adding a separate case for these operators to indicate that this is deliberate
			return nil, fmt.Errorf("%q isn't supported in label selectors", req.Operator())
		default:
			return nil, fmt.Errorf("%q is not a valid label selector operator", req.Operator())
		}
		labelSelector.MatchExpressions = append(labelSelector.MatchExpressions, LabelSelectorRequirement{
			Key:      req.Key(),
			Operator: op,
			Values:   req.Values().List(),
		})
	}
	return labelSelector, nil
}

// SetAsLabelSelector converts the labels.Set object into a LabelSelector api object.
func SetAsLabelSelector(ls labels.Set) *LabelSelector {
	if ls == nil {
		return nil
	}

	selector := &LabelSelector{
		MatchLabels: make(map[string]string, len(ls)),
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

func ExtractGroupVersions(l *APIGroupList) []string {
	var groupVersions []string
	for _, g := range l.Groups {
		for _, gv := range g.Versions {
			groupVersions = append(groupVersions, gv.GroupVersion)
		}
	}
	return groupVersions
}

// HasAnnotation returns a bool if passed in annotation exists
func HasAnnotation(obj ObjectMeta, ann string) bool {
	_, found := obj.Annotations[ann]
	return found
}

// SetMetaDataAnnotation sets the annotation and value
func SetMetaDataAnnotation(obj *ObjectMeta, ann string, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[ann] = value
}

// HasLabel returns a bool if passed in label exists
func HasLabel(obj ObjectMeta, label string) bool {
	_, found := obj.Labels[label]
	return found
}

// SetMetaDataLabel sets the label and value
func SetMetaDataLabel(obj *ObjectMeta, label string, value string) {
	if obj.Labels == nil {
		obj.Labels = make(map[string]string)
	}
	obj.Labels[label] = value
}

// SingleObject returns a ListOptions for watching a single object.
func SingleObject(meta ObjectMeta) ListOptions {
	return ListOptions{
		FieldSelector:   fields.OneTermEqualSelector("metadata.name", meta.Name).String(),
		ResourceVersion: meta.ResourceVersion,
	}
}

// NewDeleteOptions returns a DeleteOptions indicating the resource should
// be deleted within the specified grace period. Use zero to indicate
// immediate deletion. If you would prefer to use the default grace period,
// use &metav1.DeleteOptions{} directly.
func NewDeleteOptions(grace int64) *DeleteOptions {
	return &DeleteOptions{GracePeriodSeconds: &grace}
}

// NewPreconditionDeleteOptions returns a DeleteOptions with a UID precondition set.
func NewPreconditionDeleteOptions(uid string) *DeleteOptions {
	u := types.UID(uid)
	p := Preconditions{UID: &u}
	return &DeleteOptions{Preconditions: &p}
}

// NewUIDPreconditions returns a Preconditions with UID set.
func NewUIDPreconditions(uid string) *Preconditions {
	u := types.UID(uid)
	return &Preconditions{UID: &u}
}

// NewRVDeletionPrecondition returns a DeleteOptions with a ResourceVersion precondition set.
func NewRVDeletionPrecondition(rv string) *DeleteOptions {
	p := Preconditions{ResourceVersion: &rv}
	return &DeleteOptions{Preconditions: &p}
}

// HasObjectMetaSystemFieldValues returns true if fields that are managed by the system on ObjectMeta have values.
func HasObjectMetaSystemFieldValues(meta Object) bool {
	return !meta.GetCreationTimestamp().Time.IsZero() ||
		len(meta.GetUID()) != 0
}

// ResetObjectMetaForStatus forces the meta fields for a status update to match the meta fields
// for a pre-existing object. This is opt-in for new objects with Status subresource.
func ResetObjectMetaForStatus(meta, existingMeta Object) {
	meta.SetDeletionTimestamp(existingMeta.GetDeletionTimestamp())
	meta.SetGeneration(existingMeta.GetGeneration())
	meta.SetSelfLink(existingMeta.GetSelfLink())
	meta.SetLabels(existingMeta.GetLabels())
	meta.SetAnnotations(existingMeta.GetAnnotations())
	meta.SetFinalizers(existingMeta.GetFinalizers())
	meta.SetOwnerReferences(existingMeta.GetOwnerReferences())
	// managedFields must be preserved since it's been modified to
	// track changed fields in the status update.
	//meta.SetManagedFields(existingMeta.GetManagedFields())
}

// MarshalJSON implements json.Marshaler
// MarshalJSON may get called on pointers or values, so implement MarshalJSON on value.
// http://stackoverflow.com/questions/21390979/custom-marshaljson-never-gets-called-in-go
func (f FieldsV1) MarshalJSON() ([]byte, error) {
	if f.Raw == nil {
		return []byte("null"), nil
	}
	if f.getContentType() == fieldsV1InvalidOrValidCBORObject {
		var u map[string]interface{}
		if err := cbor.Unmarshal(f.Raw, &u); err != nil {
			return nil, fmt.Errorf("metav1.FieldsV1 cbor invalid: %w", err)
		}
		return utiljson.Marshal(u)
	}
	return f.Raw, nil
}

// UnmarshalJSON implements json.Unmarshaler
func (f *FieldsV1) UnmarshalJSON(b []byte) error {
	if f == nil {
		return errors.New("metav1.FieldsV1: UnmarshalJSON on nil pointer")
	}
	if !bytes.Equal(b, []byte("null")) {
		f.Raw = append(f.Raw[0:0], b...)
	}
	return nil
}

var _ json.Marshaler = FieldsV1{}
var _ json.Unmarshaler = &FieldsV1{}

func (f FieldsV1) MarshalCBOR() ([]byte, error) {
	if f.Raw == nil {
		return cbor.Marshal(nil)
	}
	if f.getContentType() == fieldsV1InvalidOrValidJSONObject {
		var u map[string]interface{}
		if err := utiljson.Unmarshal(f.Raw, &u); err != nil {
			return nil, fmt.Errorf("metav1.FieldsV1 json invalid: %w", err)
		}
		return cbor.Marshal(u)
	}
	return f.Raw, nil
}

var cborNull = []byte{0xf6}

func (f *FieldsV1) UnmarshalCBOR(b []byte) error {
	if f == nil {
		return errors.New("metav1.FieldsV1: UnmarshalCBOR on nil pointer")
	}
	if !bytes.Equal(b, cborNull) {
		f.Raw = append(f.Raw[0:0], b...)
	}
	return nil
}

const (
	// fieldsV1InvalidOrEmpty indicates that a FieldsV1 either contains no raw bytes or its raw
	// bytes don't represent an allowable value in any supported encoding.
	fieldsV1InvalidOrEmpty = iota

	// fieldsV1InvalidOrValidJSONObject indicates that a FieldV1 either contains raw bytes that
	// are a valid JSON encoding of an allowable value or don't represent an allowable value in
	// any supported encoding.
	fieldsV1InvalidOrValidJSONObject

	// fieldsV1InvalidOrValidCBORObject indicates that a FieldV1 either contains raw bytes that
	// are a valid CBOR encoding of an allowable value or don't represent an allowable value in
	// any supported encoding.
	fieldsV1InvalidOrValidCBORObject
)

// getContentType returns one of fieldsV1InvalidOrEmpty, fieldsV1InvalidOrValidJSONObject,
// fieldsV1InvalidOrValidCBORObject based on the value of Raw.
//
// Raw can be encoded in JSON or CBOR and is only valid if it is empty, null, or an object (map)
// value. It is invalid if it contains a JSON string, number, boolean, or array. If Raw is nonempty
// and represents an allowable value, then the initial byte unambiguously distinguishes a
// JSON-encoded value from a CBOR-encoded value.
//
// A valid JSON-encoded value can begin with any of the four JSON whitespace characters, the first
// character 'n' of null, or '{' (0x09, 0x0a, 0x0d, 0x20, 0x6e, or 0x7b, respectively). A valid
// CBOR-encoded value can begin with the null simple value, an initial byte with major type "map",
// or, if a tag-enclosed map, an initial byte with major type "tag" (0xf6, 0xa0...0xbf, or
// 0xc6...0xdb). The two sets of valid initial bytes don't intersect.
func (f FieldsV1) getContentType() int {
	if len(f.Raw) > 0 {
		p := f.Raw[0]
		switch p {
		case 'n', '{', '\t', '\r', '\n', ' ':
			return fieldsV1InvalidOrValidJSONObject
		case 0xf6: // null
			return fieldsV1InvalidOrValidCBORObject
		default:
			if p >= 0xa0 && p <= 0xbf /* map */ || p >= 0xc6 && p <= 0xdb /* tag */ {
				return fieldsV1InvalidOrValidCBORObject
			}
		}
	}
	return fieldsV1InvalidOrEmpty
}
