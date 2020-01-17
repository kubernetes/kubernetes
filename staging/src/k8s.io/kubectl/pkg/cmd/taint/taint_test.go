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

package taint

import (
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

func generateNodeAndTaintedNode(oldTaints []corev1.Taint, newTaints []corev1.Taint) (*corev1.Node, *corev1.Node) {
	var taintedNode *corev1.Node

	// Create a node.
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node-name",
			CreationTimestamp: metav1.Time{Time: time.Now()},
		},
		Spec: corev1.NodeSpec{
			Taints: oldTaints,
		},
		Status: corev1.NodeStatus{},
	}

	// A copy of the same node, but tainted.
	taintedNode = node.DeepCopy()
	taintedNode.Spec.Taints = newTaints

	return node, taintedNode
}

func equalTaints(taintsA, taintsB []corev1.Taint) bool {
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
		oldTaints   []corev1.Taint
		newTaints   []corev1.Taint
		args        []string
		expectFatal bool
		expectTaint bool
	}{
		// success cases
		{
			description: "taints a node with effect NoSchedule",
			newTaints: []corev1.Taint{{
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
			newTaints: []corev1.Taint{{
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
			oldTaints: []corev1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			newTaints: []corev1.Taint{{
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
			newTaints: []corev1.Taint{{
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
			oldTaints: []corev1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "PreferNoSchedule",
			}},
			newTaints: []corev1.Taint{{
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
			oldTaints: []corev1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "PreferNoSchedule",
			}},
			newTaints:   []corev1.Taint{},
			args:        []string{"node", "node-name", "dedicated-"},
			expectFatal: false,
			expectTaint: true,
		},
		{
			description: "node has two taints, update one of them and remove the other",
			oldTaints: []corev1.Taint{{
				Key:    "dedicated",
				Value:  "namespaceA",
				Effect: "NoSchedule",
			}, {
				Key:    "foo",
				Value:  "bar",
				Effect: "PreferNoSchedule",
			}},
			newTaints: []corev1.Taint{{
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
			oldTaints: []corev1.Taint{{
				Key:    "foo",
				Value:  "bar",
				Effect: "NoSchedule",
			}},
			newTaints: []corev1.Taint{{
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
		t.Run(test.description, func(t *testing.T) {
			oldNode, expectNewNode := generateNodeAndTaintedNode(test.oldTaints, test.newTaints)
			newNode := &corev1.Node{}
			tainted := false
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				GroupVersion:         corev1.SchemeGroupVersion,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					m := &MyReq{req}
					switch {
					case m.isFor("GET", "/nodes"):
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, oldNode)}, nil
					case m.isFor("GET", "/nodes/node-name"):
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, oldNode)}, nil
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
						appliedPatch, err := strategicpatch.StrategicMergePatch(oldJSON, data, &corev1.Node{})
						if err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}

						// decode the patch
						if err := runtime.DecodeInto(codec, appliedPatch, newNode); err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						if !equalTaints(expectNewNode.Spec.Taints, newNode.Spec.Taints) {
							t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, expectNewNode.Spec.Taints, newNode.Spec.Taints)
						}
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, newNode)}, nil
					case m.isFor("PUT", "/nodes/node-name"):
						tainted = true
						data, err := ioutil.ReadAll(req.Body)
						if err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						defer req.Body.Close()
						if err := runtime.DecodeInto(codec, data, newNode); err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						if !equalTaints(expectNewNode.Spec.Taints, newNode.Spec.Taints) {
							t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, expectNewNode.Spec.Taints, newNode.Spec.Taints)
						}
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, newNode)}, nil
					default:
						t.Fatalf("%s: unexpected request: %v %#v\n%#v", test.description, req.Method, req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			cmd := NewCmdTaint(tf, genericclioptions.NewTestIOStreamsDiscard())

			sawFatal := false
			func() {
				defer func() {
					// Recover from the panic below.
					if r := recover(); r != nil {
						t.Logf("Recovered: %v", r)
					}

					// Restore cmdutil behavior
					cmdutil.DefaultBehaviorOnFatal()
				}()
				cmdutil.BehaviorOnFatal(func(e string, code int) { sawFatal = true; panic(e) })
				cmd.SetArgs(test.args)
				cmd.Execute()
			}()

			if test.expectFatal {
				if !sawFatal {
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
		})
	}
}

func TestValidateFlags(t *testing.T) {
	tests := []struct {
		taintOpts   TaintOptions
		description string
		expectFatal bool
	}{

		{
			taintOpts:   TaintOptions{selector: "myLabel=X", all: false},
			description: "With Selector and without All flag",
			expectFatal: false,
		},
		{
			taintOpts:   TaintOptions{selector: "", all: true},
			description: "Without selector and All flag",
			expectFatal: false,
		},
		{
			taintOpts:   TaintOptions{selector: "myLabel=X", all: true},
			description: "With Selector and with All flag",
			expectFatal: true,
		},
		{
			taintOpts:   TaintOptions{selector: "", all: false, resources: []string{"node"}},
			description: "Without Selector and All flags and if node name is not provided",
			expectFatal: true,
		},
		{
			taintOpts:   TaintOptions{selector: "", all: false, resources: []string{"node", "node-name"}},
			description: "Without Selector and ALL flags and if node name is provided",
			expectFatal: false,
		},
	}
	for _, test := range tests {
		sawFatal := false
		err := test.taintOpts.validateFlags()
		if err != nil {
			sawFatal = true
		}
		if test.expectFatal {
			if !sawFatal {
				t.Fatalf("%s expected not to fail", test.description)
			}
		}
	}
}

type MyReq struct {
	Request *http.Request
}

func (m *MyReq) isFor(method string, path string) bool {
	req := m.Request

	return method == req.Method && (req.URL.Path == path ||
		req.URL.Path == strings.Join([]string{"/api/v1", path}, "") ||
		req.URL.Path == strings.Join([]string{"/apis/extensions/v1beta1", path}, "") ||
		req.URL.Path == strings.Join([]string{"/apis/batch/v1", path}, ""))
}
