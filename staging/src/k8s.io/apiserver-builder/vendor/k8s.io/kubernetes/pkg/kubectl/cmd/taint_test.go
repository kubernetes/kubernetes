/*
Copyright 2014 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func generateNodeAndTaintedNode(oldTaints []v1.Taint, newTaints []v1.Taint) (*v1.Node, *v1.Node) {
	var taintedNode *v1.Node

	// Create a node.
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node-name",
			CreationTimestamp: metav1.Time{Time: time.Now()},
		},
		Spec: v1.NodeSpec{
			ExternalID: "node-name",
			Taints:     oldTaints,
		},
		Status: v1.NodeStatus{},
	}
	clone, _ := api.Scheme.DeepCopy(node)

	// A copy of the same node, but tainted.
	taintedNode = clone.(*v1.Node)
	taintedNode.Spec.Taints = newTaints

	return node, taintedNode
}

func equalTaints(taintsA, taintsB []v1.Taint) bool {
	if len(taintsA) != len(taintsB) {
		return false
	}

	for _, taintA := range taintsA {
		found := false
		for _, taintB := range taintsB {
			if reflect.DeepEqual(taintA, taintB) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func TestTaint(t *testing.T) {
	tests := []struct {
		description string
		oldTaints   []v1.Taint
		newTaints   []v1.Taint
		args        []string
		expectFatal bool
		expectTaint bool
	}{
		// success cases
		{
			description: "taints a node with effect NoSchedule",
			newTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			args:        []string{"node", "node-name", "foo=bar:NoSchedule"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "taints a node with effect PreferNoSchedule",
			newTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "PreferNoSchedule",
			}},
			args:        []string{"node", "node-name", "foo=bar:PreferNoSchedule"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "update an existing taint on the node, change the value from bar to barz",
			oldTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			newTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "barz",
				Effect: "NoSchedule",
			}},
			args:        []string{"node", "node-name", "foo=barz:NoSchedule", "--overwrite"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "taints a node with two taints",
			newTaints: []v1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "foo",
				Value:  "bar",
				Effect: "PreferNoSchedule",
			}},
			args:        []string{"node", "node-name", "dedicated=namespaceA:NoSchedule", "foo=bar:PreferNoSchedule"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "node has two taints with the same key but different effect, remove one of them by indicating exact key and effect",
			oldTaints: []v1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "PreferNoSchedule",
			}},
			newTaints: []v1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "PreferNoSchedule",
			}},
			args:        []string{"node", "node-name", "dedicated:NoSchedule-"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "node has two taints with the same key but different effect, remove all of them with wildcard",
			oldTaints: []v1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "PreferNoSchedule",
			}},
			newTaints:   []v1.Taint{},
			args:        []string{"node", "node-name", "dedicated-"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "node has two taints, update one of them and remove the other",
			oldTaints: []v1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "foo",
				Value:  "bar",
				Effect: "PreferNoSchedule",
			}},
			newTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "barz",
				Effect: "PreferNoSchedule",
			}},
			args:        []string{"node", "node-name", "dedicated:NoSchedule-", "foo=barz:PreferNoSchedule", "--overwrite"},
			expectFatal: false,
			expectTaint: true,
		},

		// error cases
		{
			description: "invalid taint key",
			args:        []string{"node", "node-name", "nospecialchars^@=banana:NoSchedule"},
			expectFatal: true,
			expectTaint: false,
		},
		{
			description: "invalid taint effect",
			args:        []string{"node", "node-name", "foo=bar:NoExcute"},
			expectFatal: true,
			expectTaint: false,
		},
		{
			description: "duplicated taints with the same key and effect should be rejected",
			args:        []string{"node", "node-name", "foo=bar:NoExcute", "foo=barz:NoExcute"},
			expectFatal: true,
			expectTaint: false,
		},
		{
			description: "can't update existing taint on the node, since 'overwrite' flag is not set",
			oldTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			newTaints: []v1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			args:        []string{"node", "node-name", "foo=bar:NoSchedule"},
			expectFatal: true,
			expectTaint: false,
		},
	}

	for _, test := range tests {
		oldNode, expectNewNode := generateNodeAndTaintedNode(test.oldTaints, test.newTaints)

		new_node := &v1.Node{}
		tainted := false
		f, tf, codec, ns := cmdtesting.NewAPIFactory()

		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				m := &MyReq{req}
				switch {
				case m.isFor("GET", "/nodes/node-name"):
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, oldNode)}, nil
				case m.isFor("PATCH", "/nodes/node-name"):
					tainted = true
					data, err := ioutil.ReadAll(req.Body)
					if err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					defer req.Body.Close()

					// apply the patch
					oldJSON, err := runtime.Encode(codec, oldNode)
					if err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					appliedPatch, err := strategicpatch.StrategicMergePatch(oldJSON, data, &v1.Node{})
					if err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}

					// decode the patch
					if err := runtime.DecodeInto(codec, appliedPatch, new_node); err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					if !equalTaints(expectNewNode.Spec.Taints, new_node.Spec.Taints) {
						t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, expectNewNode.Spec.Taints, new_node.Spec.Taints)
					}
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, new_node)}, nil
				case m.isFor("PUT", "/nodes/node-name"):
					tainted = true
					data, err := ioutil.ReadAll(req.Body)
					if err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					defer req.Body.Close()
					if err := runtime.DecodeInto(codec, data, new_node); err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					if !equalTaints(expectNewNode.Spec.Taints, new_node.Spec.Taints) {
						t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, expectNewNode.Spec.Taints, new_node.Spec.Taints)
					}
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, new_node)}, nil
				default:
					t.Fatalf("%s: unexpected request: %v %#v\n%#v", test.description, req.Method, req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.ClientConfig = defaultClientConfig()

		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdTaint(f, buf)

		saw_fatal := false
		func() {
			defer func() {
				// Recover from the panic below.
				_ = recover()
				// Restore cmdutil behavior
				cmdutil.DefaultBehaviorOnFatal()
			}()
			cmdutil.BehaviorOnFatal(func(e string, code int) { saw_fatal = true; panic(e) })
			cmd.SetArgs(test.args)
			cmd.Execute()
		}()

		if test.expectFatal {
			if !saw_fatal {
				t.Fatalf("%s: unexpected non-error", test.description)
			}
		}

		if test.expectTaint {
			if !tainted {
				t.Fatalf("%s: node not tainted", test.description)
			}
		}
		if !test.expectTaint {
			if tainted {
				t.Fatalf("%s: unexpected taint", test.description)
			}
		}
	}
}
