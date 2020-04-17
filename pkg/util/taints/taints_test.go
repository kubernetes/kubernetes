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

	"k8s.io/api/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"

	"github.com/spf13/pflag"
)

func TestTaintsVar(t *testing.T) {
	cases := []struct {
		f   string
		err bool
		t   []api.Taint
	}{
		{
			f: "",
			t: []api.Taint(nil),
		},
		{
			f: "--t=foo=bar:NoSchedule",
			t: []api.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}},
		},
		{
			f: "--t=baz:NoSchedule",
			t: []api.Taint{{Key: "baz", Value: "", Effect: "NoSchedule"}},
		},
		{
			f: "--t=foo=bar:NoSchedule,baz:NoSchedule,bing=bang:PreferNoSchedule,qux=:NoSchedule",
			t: []api.Taint{
				{Key: "foo", Value: "bar", Effect: api.TaintEffectNoSchedule},
				{Key: "baz", Value: "", Effect: "NoSchedule"},
				{Key: "bing", Value: "bang", Effect: api.TaintEffectPreferNoSchedule},
				{Key: "qux", Value: "", Effect: "NoSchedule"},
			},
		},
		{
			f: "--t=dedicated-for=user1:NoExecute,baz:NoSchedule,foo-bar=:NoSchedule",
			t: []api.Taint{
				{Key: "dedicated-for", Value: "user1", Effect: "NoExecute"},
				{Key: "baz", Value: "", Effect: "NoSchedule"},
				{Key: "foo-bar", Value: "", Effect: "NoSchedule"},
			},
		},
	}

	for i, c := range cases {
		args := append([]string{"test"}, strings.Fields(c.f)...)
		cli := pflag.NewFlagSet("test", pflag.ContinueOnError)
		var taints []api.Taint
		cli.Var(NewTaintsVar(&taints), "t", "bar")

		err := cli.Parse(args)
		if err == nil && c.err {
			t.Errorf("[%v] expected error", i)
			continue
		}
		if err != nil && !c.err {
			t.Errorf("[%v] unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(c.t, taints) {
			t.Errorf("[%v] unexpected taints:\n\texpected:\n\t\t%#v\n\tgot:\n\t\t%#v", i, c.t, taints)
		}
	}

}

func TestAddOrUpdateTaint(t *testing.T) {
	node := &v1.Node{}

	taint := &v1.Taint{
		Key:    "foo",
		Value:  "bar",
		Effect: v1.TaintEffectNoSchedule,
	}

	checkResult := func(testCaseName string, newNode *v1.Node, expectedTaint *v1.Taint, result, expectedResult bool, err error) {
		if err != nil {
			t.Errorf("[%s] should not raise error but got %v", testCaseName, err)
		}
		if result != expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", testCaseName, expectedResult, result)
		}
		if len(newNode.Spec.Taints) != 1 || !reflect.DeepEqual(newNode.Spec.Taints[0], *expectedTaint) {
			t.Errorf("[%s] node should only have one taint: %v, but got: %v", testCaseName, *expectedTaint, newNode.Spec.Taints)
		}
	}

	// Add a new Taint.
	newNode, result, err := AddOrUpdateTaint(node, taint)
	checkResult("Add New Taint", newNode, taint, result, true, err)

	// Update a Taint.
	taint.Value = "bar_1"
	newNode, result, err = AddOrUpdateTaint(node, taint)
	checkResult("Update Taint", newNode, taint, result, true, err)

	// Add a duplicate Taint.
	node = newNode
	newNode, result, err = AddOrUpdateTaint(node, taint)
	checkResult("Add Duplicate Taint", newNode, taint, result, false, err)
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

func TestReorganizeTaints(t *testing.T) {
	node := &v1.Node{
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{
				{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	cases := []struct {
		name              string
		overwrite         bool
		taintsToAdd       []v1.Taint
		taintsToDelete    []v1.Taint
		expectedTaints    []v1.Taint
		expectedOperation string
		expectedErr       bool
	}{
		{
			name:              "no changes with overwrite is true",
			overwrite:         true,
			taintsToAdd:       []v1.Taint{},
			taintsToDelete:    []v1.Taint{},
			expectedTaints:    node.Spec.Taints,
			expectedOperation: MODIFIED,
			expectedErr:       false,
		},
		{
			name:              "no changes with overwrite is false",
			overwrite:         false,
			taintsToAdd:       []v1.Taint{},
			taintsToDelete:    []v1.Taint{},
			expectedTaints:    node.Spec.Taints,
			expectedOperation: UNTAINTED,
			expectedErr:       false,
		},
		{
			name:      "add new taint",
			overwrite: false,
			taintsToAdd: []v1.Taint{
				{
					Key:    "foo_1",
					Effect: v1.TaintEffectNoExecute,
				},
			},
			taintsToDelete:    []v1.Taint{},
			expectedTaints:    append([]v1.Taint{{Key: "foo_1", Effect: v1.TaintEffectNoExecute}}, node.Spec.Taints...),
			expectedOperation: TAINTED,
			expectedErr:       false,
		},
		{
			name:        "delete taint with effect",
			overwrite:   false,
			taintsToAdd: []v1.Taint{},
			taintsToDelete: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaints:    []v1.Taint{},
			expectedOperation: UNTAINTED,
			expectedErr:       false,
		},
		{
			name:        "delete taint with no effect",
			overwrite:   false,
			taintsToAdd: []v1.Taint{},
			taintsToDelete: []v1.Taint{
				{
					Key: "foo",
				},
			},
			expectedTaints:    []v1.Taint{},
			expectedOperation: UNTAINTED,
			expectedErr:       false,
		},
		{
			name:        "delete non-exist taint",
			overwrite:   false,
			taintsToAdd: []v1.Taint{},
			taintsToDelete: []v1.Taint{
				{
					Key:    "foo_1",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaints:    node.Spec.Taints,
			expectedOperation: UNTAINTED,
			expectedErr:       true,
		},
		{
			name:      "add new taint and delete old one",
			overwrite: false,
			taintsToAdd: []v1.Taint{
				{
					Key:    "foo_1",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintsToDelete: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo_1",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedOperation: MODIFIED,
			expectedErr:       false,
		},
	}

	for _, c := range cases {
		operation, taints, err := ReorganizeTaints(node, c.overwrite, c.taintsToAdd, c.taintsToDelete)
		if c.expectedErr && err == nil {
			t.Errorf("[%s] expect to see an error, but did not get one", c.name)
		} else if !c.expectedErr && err != nil {
			t.Errorf("[%s] expect not to see an error, but got one: %v", c.name, err)
		}

		if !reflect.DeepEqual(c.expectedTaints, taints) {
			t.Errorf("[%s] expect to see taint list %#v, but got: %#v", c.name, c.expectedTaints, taints)
		}

		if c.expectedOperation != operation {
			t.Errorf("[%s] expect to see operation %s, but got: %s", c.name, c.expectedOperation, operation)
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
			name:        "invalid spec format",
			spec:        []string{""},
			expectedErr: true,
		},
		{
			name:        "invalid spec format",
			spec:        []string{"foo=abc"},
			expectedErr: true,
		},
		{
			name:        "invalid spec format",
			spec:        []string{"foo=abc=xyz:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec format",
			spec:        []string{"foo=abc:xyz:NoSchedule"},
			expectedErr: true,
		},
		{
			name:        "invalid spec format for adding taint",
			spec:        []string{"foo"},
			expectedErr: true,
		},
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
			name: "add new taints",
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
			name: "delete taints",
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
			name: "add taints and delete taints",
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
			t.Errorf("[%s] expected returen taints as %v, but got: %v", c.name, c.expectedTaints, taints)
		}
		if !reflect.DeepEqual(c.expectedTaintsToRemove, taintsToRemove) {
			t.Errorf("[%s] expected return taints to be removed as %v, but got: %v", c.name, c.expectedTaintsToRemove, taintsToRemove)
		}
	}
}
