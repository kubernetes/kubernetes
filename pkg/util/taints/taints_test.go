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
	"k8s.io/kubernetes/pkg/api"

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
			f: "--t=foo=bar:NoSchedule,bing=bang:PreferNoSchedule",
			t: []api.Taint{
				{Key: "foo", Value: "bar", Effect: api.TaintEffectNoSchedule},
				{Key: "bing", Value: "bang", Effect: api.TaintEffectPreferNoSchedule},
			},
		},
		{
			f: "--t=dedicated-for=user1:NoExecute",
			t: []api.Taint{{Key: "dedicated-for", Value: "user1", Effect: "NoExecute"}},
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
