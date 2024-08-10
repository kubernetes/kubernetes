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

package taints

import (
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"

	"github.com/google/go-cmp/cmp"
)

func TestAddOrUpdateTaint(t *testing.T) {
	taint := v1.Taint{
		Key:    "foo",
		Value:  "bar",
		Effect: v1.TaintEffectNoSchedule,
	}

	taintNew := v1.Taint{
		Key:    "foo_1",
		Value:  "bar_1",
		Effect: v1.TaintEffectNoSchedule,
	}

	taintUpdateValue := taint
	taintUpdateValue.Value = "bar_1"

	testcases := []struct {
		name           string
		node           *v1.Node
		taint          *v1.Taint
		expectedUpdate bool
		expectedTaints []v1.Taint
	}{
		{
			name:           "add a new taint",
			node:           &v1.Node{},
			taint:          &taint,
			expectedUpdate: true,
			expectedTaints: []v1.Taint{taint},
		},
		{
			name: "add a unique taint",
			node: &v1.Node{
				Spec: v1.NodeSpec{Taints: []v1.Taint{taint}},
			},
			taint:          &taintNew,
			expectedUpdate: true,
			expectedTaints: []v1.Taint{taint, taintNew},
		},
		{
			name: "add duplicate taint",
			node: &v1.Node{
				Spec: v1.NodeSpec{Taints: []v1.Taint{taint}},
			},
			taint:          &taint,
			expectedUpdate: false,
			expectedTaints: []v1.Taint{taint},
		},
		{
			name: "update taint value",
			node: &v1.Node{
				Spec: v1.NodeSpec{Taints: []v1.Taint{taint}},
			},
			taint:          &taintUpdateValue,
			expectedUpdate: true,
			expectedTaints: []v1.Taint{taintUpdateValue},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			newNode, updated, err := AddOrUpdateTaint(tc.node, tc.taint)
			if err != nil {
				t.Errorf("[%s] should not raise error but got %v", tc.name, err)
			}
			if updated != tc.expectedUpdate {
				t.Errorf("[%s] expected taints to not be updated", tc.name)
			}
			if diff := cmp.Diff(newNode.Spec.Taints, tc.expectedTaints); diff != "" {
				t.Errorf("Unexpected result (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestTaintExists(t *testing.T) {
	testingTaints := []v1.Taint{
		{
			Key:    "foo_1",
			Value:  "bar_1",
			Effect: v1.TaintEffectNoExecute,
		},
		{
			Key:    "foo_2",
			Value:  "bar_2",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	cases := []struct {
		name           string
		taintToFind    *v1.Taint
		expectedResult bool
	}{
		{
			name:           "taint exists",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			expectedResult: true,
		},
		{
			name:           "different key",
			taintToFind:    &v1.Taint{Key: "no_such_key", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			expectedResult: false,
		},
		{
			name:           "different effect",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		result := TaintExists(testingTaints, c.taintToFind)

		if result != c.expectedResult {
			t.Errorf("[%s] unexpected results: %v", c.name, result)
			continue
		}
	}
}

func TestTaintKeyExists(t *testing.T) {
	testingTaints := []v1.Taint{
		{
			Key:    "foo_1",
			Value:  "bar_1",
			Effect: v1.TaintEffectNoExecute,
		},
		{
			Key:    "foo_2",
			Value:  "bar_2",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	cases := []struct {
		name            string
		taintKeyToMatch string
		expectedResult  bool
	}{
		{
			name:            "taint key exists",
			taintKeyToMatch: "foo_1",
			expectedResult:  true,
		},
		{
			name:            "taint key does not exist",
			taintKeyToMatch: "foo_3",
			expectedResult:  false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := TaintKeyExists(testingTaints, c.taintKeyToMatch)

			if result != c.expectedResult {
				t.Errorf("[%s] unexpected results: %v", c.name, result)
			}
		})
	}
}

func TestTaintSetFilter(t *testing.T) {
	testTaint1 := v1.Taint{
		Key:    "foo_1",
		Value:  "bar_1",
		Effect: v1.TaintEffectNoExecute,
	}
	testTaint2 := v1.Taint{
		Key:    "foo_2",
		Value:  "bar_2",
		Effect: v1.TaintEffectNoSchedule,
	}

	testTaint3 := v1.Taint{
		Key:    "foo_3",
		Value:  "bar_3",
		Effect: v1.TaintEffectNoSchedule,
	}
	testTaints := []v1.Taint{testTaint1, testTaint2, testTaint3}

	testcases := []struct {
		name           string
		fn             func(t *v1.Taint) bool
		expectedTaints []v1.Taint
	}{
		{
			name: "Filter out nothing",
			fn: func(t *v1.Taint) bool {
				if t.Key == v1.TaintNodeUnschedulable {
					return true
				}
				return false
			},
			expectedTaints: []v1.Taint{},
		},
		{
			name: "Filter out a subset",
			fn: func(t *v1.Taint) bool {
				if t.Effect == v1.TaintEffectNoExecute {
					return true
				}
				return false
			},
			expectedTaints: []v1.Taint{testTaint1},
		},
		{
			name:           "Filter out everything",
			fn:             func(t *v1.Taint) bool { return true },
			expectedTaints: []v1.Taint{testTaint1, testTaint2, testTaint3},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			taintsAfterFilter := TaintSetFilter(testTaints, tc.fn)
			if diff := cmp.Diff(tc.expectedTaints, taintsAfterFilter); diff != "" {
				t.Errorf("Unexpected postFilterResult (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestRemoveTaint(t *testing.T) {
	cases := []struct {
		name           string
		node           *v1.Node
		taintToRemove  *v1.Taint
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "remove taint unsuccessfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo_1",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "remove taint successfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name: "remove taint from node with no taint",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		newNode, result, err := RemoveTaint(c.node, c.taintToRemove)
		if err != nil {
			t.Errorf("[%s] should not raise error but got: %v", c.name, err)
		}
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(newNode.Spec.Taints, c.expectedTaints) {
			t.Errorf("[%s] the new node object should have taints %v, but got: %v", c.name, c.expectedTaints, newNode.Spec.Taints)
		}
	}
}

func TestDeleteTaint(t *testing.T) {
	cases := []struct {
		name           string
		taints         []v1.Taint
		taintToDelete  *v1.Taint
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "delete taint with different name",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo_1", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint with different effect",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoExecute},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint successfully",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name:           "delete taint from empty taint array",
			taints:         []v1.Taint{},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		taints, result := DeleteTaint(c.taints, c.taintToDelete)
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(taints, c.expectedTaints) {
			t.Errorf("[%s] the result taints should be %v, but got: %v", c.name, c.expectedTaints, taints)
		}
	}
}

func TestDeleteTaintByKey(t *testing.T) {
	cases := []struct {
		name           string
		taints         []v1.Taint
		taintKey       string
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "delete taint unsuccessfully",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintKey: "foo_1",
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint successfully",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintKey:       "foo",
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name:           "delete taint from empty taint array",
			taints:         []v1.Taint{},
			taintKey:       "foo",
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		taints, result := DeleteTaintsByKey(c.taints, c.taintKey)
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(c.expectedTaints, taints) {
			t.Errorf("[%s] the result taints should be %v, but got: %v", c.name, c.expectedTaints, taints)
		}
	}
}

func TestCheckIfTaintsAlreadyExists(t *testing.T) {
	oldTaints := []v1.Taint{
		{
			Key:    "foo_1",
			Value:  "bar",
			Effect: v1.TaintEffectNoSchedule,
		},
		{
			Key:    "foo_2",
			Value:  "bar",
			Effect: v1.TaintEffectNoSchedule,
		},
		{
			Key:    "foo_3",
			Value:  "bar",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	cases := []struct {
		name           string
		taintsToCheck  []v1.Taint
		expectedResult string
	}{
		{
			name:           "empty array",
			taintsToCheck:  []v1.Taint{},
			expectedResult: "",
		},
		{
			name: "no match",
			taintsToCheck: []v1.Taint{
				{
					Key:    "foo_1",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			expectedResult: "",
		},
		{
			name: "match one taint",
			taintsToCheck: []v1.Taint{
				{
					Key:    "foo_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: "foo_2",
		},
		{
			name: "match two taints",
			taintsToCheck: []v1.Taint{
				{
					Key:    "foo_2",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "foo_3",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: "foo_2,foo_3",
		},
	}

	for _, c := range cases {
		result := CheckIfTaintsAlreadyExists(oldTaints, c.taintsToCheck)
		if result != c.expectedResult {
			t.Errorf("[%s] should return '%s', but got: '%s'", c.name, c.expectedResult, result)
		}
	}
}

func TestParseTaints(t *testing.T) {
	cases := []struct {
		name                   string
		spec                   []string
		expectedTaints         []v1.Taint
		expectedTaintsToRemove []v1.Taint
		expectedErr            bool
	}{
		{
			name:        "invalid empty spec format",
			spec:        []string{""},
			expectedErr: true,
		},
		// taint spec format without the suffix '-' must be either '<key>=<value>:<effect>', '<key>:<effect>', or '<key>'
		{
			name:        "invalid spec format without effect",
			spec:        []string{"foo=abc"},
			expectedErr: true,
		},
		{
			name:        "invalid spec format with multiple '=' separators",
			spec:        []string{"foo=abc=xyz:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec format with multiple ':' separators",
			spec:        []string{"foo=abc:xyz:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value without separator",
			spec:        []string{"foo"},
			expectedErr: true,
		},
		// taint spec must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character.
		{
			name:        "invalid spec taint value with special chars '%^@'",
			spec:        []string{"foo=nospecialchars%^@:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with non-alphanumeric characters",
			spec:        []string{"foo=Tama-nui-te-rā.is.Māori.sun:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with special chars '\\'",
			spec:        []string{"foo=\\backslashes\\are\\bad:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with start with an non-alphanumeric character '-'",
			spec:        []string{"foo=-starts-with-dash:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with end with an non-alphanumeric character '-'",
			spec:        []string{"foo=ends-with-dash-:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with start with an non-alphanumeric character '.'",
			spec:        []string{"foo=.starts.with.dot:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value with end with an non-alphanumeric character '.'",
			spec:        []string{"foo=ends.with.dot.:NoSchedule"},
			expectedErr: true,
		},
		// The value range of taint effect is "NoSchedule", "PreferNoSchedule", "NoExecute"
		{
			name:        "invalid spec effect for adding taint",
			spec:        []string{"foo=abc:invalid_effect"},
			expectedErr: true,
		},
		{
			name:        "invalid spec effect for deleting taint",
			spec:        []string{"foo:invalid_effect-"},
			expectedErr: true,
		},
		{
			name:        "duplicated taints with the same key and effect",
			spec:        []string{"foo=abc:NoSchedule", "foo=abc:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec taint value exceeding the limit",
			spec:        []string{strings.Repeat("a", 64)},
			expectedErr: true,
		},
		{
			name: "add new taints with no special chars",
			spec: []string{"foo=abc:NoSchedule", "bar=abc:NoSchedule", "baz:NoSchedule", "qux:NoSchedule", "foobar=:NoSchedule"},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "abc",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "bar",
					Value:  "abc",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "baz",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "qux",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "foobar",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedErr: false,
		},
		{
			name: "delete taints with no special chars",
			spec: []string{"foo:NoSchedule-", "bar:NoSchedule-", "qux=:NoSchedule-", "dedicated-"},
			expectedTaintsToRemove: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "qux",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key: "dedicated",
				},
			},
			expectedErr: false,
		},
		{
			name: "add taints and delete taints with no special chars",
			spec: []string{"foo=abc:NoSchedule", "bar=abc:NoSchedule", "baz:NoSchedule", "qux:NoSchedule", "foobar=:NoSchedule", "foo:NoSchedule-", "bar:NoSchedule-", "baz=:NoSchedule-"},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "abc",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "bar",
					Value:  "abc",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "baz",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "qux",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "foobar",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToRemove: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
				{
					Key:    "baz",
					Value:  "",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedErr: false,
		},
	}

	for _, c := range cases {
		taints, taintsToRemove, err := ParseTaints(c.spec)
		if c.expectedErr && err == nil {
			t.Errorf("[%s] expected error for spec %s, but got nothing", c.name, c.spec)
		}
		if !c.expectedErr && err != nil {
			t.Errorf("[%s] expected no error for spec %s, but got: %v", c.name, c.spec, err)
		}
		if !reflect.DeepEqual(c.expectedTaints, taints) {
			t.Errorf("[%s] expected return taints as %v, but got: %v", c.name, c.expectedTaints, taints)
		}
		if !reflect.DeepEqual(c.expectedTaintsToRemove, taintsToRemove) {
			t.Errorf("[%s] expected return taints to be removed as %v, but got: %v", c.name, c.expectedTaintsToRemove, taintsToRemove)
		}
	}
}

func TestValidateTaint(t *testing.T) {
	cases := []struct {
		name          string
		taintsToCheck v1.Taint
		expectedErr   bool
	}{
		{
			name:          "taint invalid key",
			taintsToCheck: v1.Taint{Key: "", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			expectedErr:   true,
		},
		{
			name:          "taint invalid value",
			taintsToCheck: v1.Taint{Key: "foo_1", Value: strings.Repeat("a", 64), Effect: v1.TaintEffectNoExecute},
			expectedErr:   true,
		},
		{
			name:          "taint invalid effect",
			taintsToCheck: v1.Taint{Key: "foo_2", Value: "bar_2", Effect: "no_such_effect"},
			expectedErr:   true,
		},
		{
			name:          "valid taint",
			taintsToCheck: v1.Taint{Key: "foo_3", Value: "bar_3", Effect: v1.TaintEffectNoExecute},
			expectedErr:   false,
		},
		{
			name:          "valid taint",
			taintsToCheck: v1.Taint{Key: "foo_4", Effect: v1.TaintEffectNoExecute},
			expectedErr:   false,
		},
		{
			name:          "valid taint",
			taintsToCheck: v1.Taint{Key: "foo_5", Value: "bar_5"},
			expectedErr:   false,
		},
	}

	for _, c := range cases {
		err := CheckTaintValidation(c.taintsToCheck)

		if c.expectedErr && err == nil {
			t.Errorf("[%s] expected error for spec %+v, but got nothing", c.name, c.taintsToCheck)
		}
	}
}

func TestTaintSetDiff(t *testing.T) {
	cases := []struct {
		name                   string
		t1                     []v1.Taint
		t2                     []v1.Taint
		expectedTaintsToAdd    []*v1.Taint
		expectedTaintsToRemove []*v1.Taint
	}{
		{
			name:                   "two_taints_are_nil",
			expectedTaintsToAdd:    nil,
			expectedTaintsToRemove: nil,
		},
		{
			name: "one_taint_is_nil_and_the_other_is_not_nil",
			t1: []v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToAdd: []*v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToRemove: nil,
		},
		{
			name: "shared_taints_with_the_same_key_value_effect",
			t1: []v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			t2: []v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToAdd: []*v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			expectedTaintsToRemove: []*v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
			},
		},
		{
			name: "shared_taints_with_the_same_key_effect_different_value",
			t1: []v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "different-value",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			t2: []v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToAdd: []*v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			expectedTaintsToRemove: []*v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
			},
		},
		{
			name: "shared_taints_with_the_same_key_different_value_effect",
			t1: []v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "different-value",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			t2: []v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaintsToAdd: []*v1.Taint{
				{
					Key:    "foo_1",
					Value:  "bar_1",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "different-value",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			expectedTaintsToRemove: []*v1.Taint{
				{
					Key:    "foo_3",
					Value:  "bar_3",
					Effect: v1.TaintEffectNoExecute,
				},
				{
					Key:    "foo_2",
					Value:  "bar_2",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			add, remove := TaintSetDiff(tt.t1, tt.t2)
			if !reflect.DeepEqual(add, tt.expectedTaintsToAdd) {
				t.Errorf("taintsToAdd: %v should equal %v, but get unexpected results", add, tt.expectedTaintsToAdd)
			}
			if !reflect.DeepEqual(remove, tt.expectedTaintsToRemove) {
				t.Errorf("taintsToRemove: %v should equal %v, but get unexpected results", remove, tt.expectedTaintsToRemove)
			}
		})
	}
}
