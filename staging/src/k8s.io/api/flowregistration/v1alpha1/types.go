/*
Copyright 2018 The Kubernetes Authors.

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

// +k8s:openapi-gen=true

package v1alpha1

import (
	"fmt"
	"net/http"
	"regexp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Level defines the amount of information logged during flowing
type FlowSource string

// Valid flow distinguishers
const (
	// LevelNone disables flowing
	FlowSourceUser FlowSource = "user"
	// LevelMetadata provides the basic level of flowing.
	FlowSourceNamespace FlowSource = "namespace"
)

// Level defines the amount of information logged during flowing
type AndField string

// Valid flow distinguishers
const (
	// LevelNone disables flowing
	AndFieldUser AndField = "user"
	// LevelMetadata provides the basic level of flowing.
	AndFieldGroups AndField = "groups"
	// LevelMetadata provides the basic level of flowing.
	AndFieldNamespace AndField = "namespace"
	// LevelMetadata provides the basic level of flowing.
	AndFieldResource AndField = "resource"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// FlowSchema represents a cluster level flow sink
type FlowSchema struct {
	metav1.TypeMeta `json:",inline" yaml:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" yaml:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the flow configuration spec
	Spec FlowSchemaSpec `json:"spec,omitempty" yaml:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

type FieldGetter interface {
	GetField(req *http.Request, field string) (bool, []string, error)
}

type MatchAll struct {
	Field string `json:"field" yaml:"field"`
}

type Equals struct {
	Field string `json:"field" yaml:"field"`
	Value string `json:"value" yaml:"value"`
}

type SuperSet struct {
	Field string   `json:"field" yaml:"field"`
	Set   []string `json:"set" yaml:"set"`
}

type PatternMatch struct {
	Field   string `json:"field" yaml:"field"`
	Pattern string `json:"pattern" yaml:"pattern"`
}

type InSet struct {
	Field string   `json:"field" yaml:"field"`
	Set   []string `json:"set" yaml:"set"`
}

type NotMatchAll struct {
	Field string `json:"field" yaml:"field"`
}

type NotEquals struct {
	Field string `json:"field" yaml:"field"`
	Value string `json:"value" yaml:"value"`
}

type NotPatternMatch struct {
	Field   string `json:"field" yaml:"field"`
	Pattern string `json:"pattern" yaml:"pattern"`
}

type NotInSet struct {
	Field string   `json:"field" yaml:"field"`
	Set   []string `json:"set" yaml:"set"`
}

type NotSuperSet struct {
	Field string   `json:"field" yaml:"field"`
	Set   []string `json:"set" yaml:"set"`
}

type Match struct {
	And *And `json:"and" yaml:"and"`
}

type And struct {
	MatchAll        MatchAll        `json:"matchAll,omitempty" yaml:"matchAll,omitempty"`
	Equals          Equals          `json:"equals,omitempty" yaml:"equals,omitempty"`
	SuperSet        SuperSet        `json:"superSet,omitempty" yaml:"superSet,omitempty"`
	PatternMatch    PatternMatch    `json:"patternMatch,omitempty" yaml:"patternMatch,omitempty"`
	InSet           InSet           `json:"inSet,omitempty" yaml:"inSet,omitempty"`
	NotEquals       NotEquals       `json:"notEquals,omitempty" yaml:"notEquals,omitempty"`
	NotPatternMatch NotPatternMatch `json:"notPatternMatch,omitempty" yaml:"notPatternMatch,omitempty"`
	NotInSet        NotInSet        `json:"notInSet,omitempty" yaml:"notInSet,omitempty"`
	NotSuperSet     NotSuperSet     `json:"notSuperSet,omitempty" yaml:"notSuperSet,omitempty"`
	NotMatchAll     NotMatchAll     `json:"notMatchAll,omitempty" yaml:"notMatchAll,omitempty"`
}

type FlowDistinguisher struct {
	Source FlowSource `json:"source" yaml:"source"`
}

type RequestPriority struct {
	Name string `json:"name" yaml:"name"`
}

// FlowSchemaSpec holds the spec for the flow sink
type FlowSchemaSpec struct {
	// TODO(aaron-prindle) add comments to fields, etc.
	RequestPriority   RequestPriority   `json:"requestPriority" yaml:"requestPriority"`
	FlowDistinguisher FlowDistinguisher `json:"flowDistinguisher" yaml:"flowDistinguisher"`
	Match             []*Match          `json:"match" yaml:"match"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// FlowSchemaList is a list of FlowSchema items.
type FlowSchemaList struct {
	metav1.TypeMeta `json:",inline" yaml:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" yaml:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of flow configurations.
	Items []FlowSchema `json:"items" yaml:"items" protobuf:"bytes,2,rep,name=items"`
}

func (ma *MatchAll) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	return true, nil
}

func (e *Equals) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	ok, fields, err := fg.GetField(req, e.Field)
	if err != nil {
		return false, err
	}
	if ok {
		return fields[0] == e.Value, nil
	}
	return false, nil
}

func (e *InSet) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	ok, fields, err := fg.GetField(req, e.Field)
	if err != nil {
		return false, err
	}
	if ok {
		for _, v := range e.Set {
			if v == fields[0] {
				return true, nil
			}
		}
	}
	return false, nil
}

// subset returns true if the first array is completely
// contained in the second array. There must be at least
// the same number of duplicate values in second as there
// are in first.
func subset(first, second []string) bool {
	set := make(map[string]int)
	for _, value := range second {
		set[value] += 1
	}

	for _, value := range first {
		if count, found := set[value]; !found {
			return false
		} else if count < 1 {
			return false
		} else {
			set[value] = count - 1
		}
	}

	return true
}

func (ss *SuperSet) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	ok, fields, err := fg.GetField(req, ss.Field)
	if err != nil {
		return false, err
	}
	if ok {
		return subset(ss.Set, fields), nil
	}
	return false, nil
}

func (pm *PatternMatch) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	ok, fields, err := fg.GetField(req, pm.Field)
	if err != nil {
		return false, err
	}
	if ok {
		match, err := regexp.MatchString(pm.Pattern, fields[0])
		if err != nil {
			return false, err
		}
		return match, nil
	}
	return false, nil
}

func (a *And) Execute(req *http.Request, fg FieldGetter) (bool, error) {
	fmt.Println("a")

	fmt.Printf("and: %v\n", a)
	// change to switch statement
	matched := true
	wasntSet := true
	if a.hasmatchall() {
		// fmt.Println("0")
		wasntSet = false
		fmatched, err := a.MatchAll.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && fmatched
	}

	if a.hasequals() {
		fmt.Println("1")

		wasntSet = false
		fmatched, err := a.Equals.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && fmatched
	}
	if a.hasnotequals() {
		fmt.Println("2")
		wasntSet = false
		equals := &Equals{
			a.NotEquals.Field,
			a.NotEquals.Value,
		}
		fmatched, err := equals.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && !fmatched // not equals
	}
	if a.hasinset() {
		fmt.Println("3")
		wasntSet = false
		fmatched, err := a.InSet.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && fmatched
	}
	if a.hasnotinset() {
		fmt.Println("4")
		wasntSet = false
		inset := &InSet{
			a.NotInSet.Field,
			a.NotInSet.Set,
		}
		fmatched, err := inset.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && !fmatched // not inset
	}
	if a.hassuperset() {
		fmt.Println("5")
		wasntSet = false
		fmatched, err := a.SuperSet.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && fmatched

	}
	if a.hasnotsuperset() {
		fmt.Println("6")
		wasntSet = false
		ss := &SuperSet{
			a.NotSuperSet.Field,
			a.NotSuperSet.Set,
		}
		fmatched, err := ss.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && !fmatched // not superset

	}
	if a.haspatternmatch() {
		fmt.Println("7")
		wasntSet = false
		fmatched, err := a.PatternMatch.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && fmatched

	}
	if a.hasnotpatternmatch() {
		fmt.Println("8")
		wasntSet = false
		pm := &PatternMatch{
			a.NotPatternMatch.Field,
			a.NotPatternMatch.Pattern,
		}
		fmatched, err := pm.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && !fmatched // not patternmatch
	}
	if a.hasnotmatchall() {
		fmt.Println("9")
		wasntSet = false
		ma := &MatchAll{
			a.NotMatchAll.Field,
		}
		fmatched, err := ma.Execute(req, fg)
		if err != nil {
			return false, err
		}
		matched = matched && !fmatched // not patternmatch
	}

	if wasntSet {
		return false, fmt.Errorf("no match directives")
	}

	return matched, nil
}

// TODO(aaron-prindle) better way to do this?
func (a *And) hasmatchall() bool {
	return a.MatchAll.Field != "" // && a.MatchAll.Value != ""
}

func (a *And) hasnotmatchall() bool {
	return a.NotMatchAll.Field != "" // && a.MatchAll.Value != ""
}

func (a *And) hasequals() bool {
	return a.Equals.Field != "" // && a.Equals.Value != ""
}

func (a *And) hasnotequals() bool {
	return a.NotEquals.Field != "" // && a.NotEquals.Value != ""
}

func (a *And) hassuperset() bool {
	return a.SuperSet.Field != "" // && a.SuperSet.Set != ""
}

func (a *And) hasnotsuperset() bool {
	return a.NotSuperSet.Field != "" // && a.SuperSet.Set != ""
}

func (a *And) hasinset() bool {
	return a.InSet.Field != "" // && a.SuperSet.Set != ""
}

func (a *And) hasnotinset() bool {
	return a.NotInSet.Field != "" // && a.SuperSet.Set != ""
}

func (a *And) haspatternmatch() bool {
	return a.PatternMatch.Field != "" // && a.SuperSet.Set != ""
}

func (a *And) hasnotpatternmatch() bool {
	return a.NotPatternMatch.Field != "" // && a.SuperSet.Set != ""
}
