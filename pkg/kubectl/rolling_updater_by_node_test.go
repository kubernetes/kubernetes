/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
)

func Node(name string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"node-label-foo": "v1"},
		},
	}
}

func Pod(name string, nodeName string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"version": "v1"},
		},
		Spec: api.PodSpec{
			NodeName: nodeName,
		},
	}
}

func oldRcByNode(replicas int, original int) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo-v1",
			UID:  "7764ae47-9092-11e4-8393-42010af018ff",
			Annotations: map[string]string{
				originalReplicasAnnotation: fmt.Sprintf("%d", original),
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{"version": "v1"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo-v1",
					Labels: map[string]string{"version": "v1"},
				},
				Spec: api.PodSpec{
					NodeSelector: map[string]string{"node-label-foo": "v1"},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: replicas,
		},
	}
}

func newRcByNode(replicas int, desired int) *api.ReplicationController {
	rc := oldRcByNode(replicas, replicas)
	rc.Spec.Template = &api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo-v2",
			Labels: map[string]string{"version": "v2"},
		},
		Spec: api.PodSpec{
			NodeSelector: map[string]string{"node-label-foo": "v2"},
		},
	}
	rc.Spec.Selector = map[string]string{"version": "v2"}
	rc.ObjectMeta = api.ObjectMeta{
		Name: "foo-v2",
		Annotations: map[string]string{
			desiredReplicasAnnotation: fmt.Sprintf("%d", desired),
			sourceIdAnnotation:        "foo-v1:7764ae47-9092-11e4-8393-42010af018ff",
		},
	}
	return rc
}

// TestUpdateByNode performs complex scenario testing for rolling updates. It
// provides fine grained control over the states for each update interval to
// allow the expression of as many edge cases as possible.
func TestUpdateByNode(t *testing.T) {
	// up represents a simulated scale up event and expectation
	type up struct {
		// to is the expected replica count for a scale-up
		to int
	}
	// down represents a simulated scale down event and expectation
	type down struct {
		// oldReady is the number of oldRcByNode replicas which will be seen
		// as ready during the scale down attempt
		oldReady int
		// newReady is the number of newRcByNode replicas which will be seen
		// as ready during the scale up attempt
		newReady int
		// to is the expected replica count for the scale down
		to int
		// noop and to are mutually exclusive; if noop is true, that means for
		// this down event, no scaling attempt should be made (for example, if
		// by scaling down, the readiness minimum would be crossed.)
		noop bool
	}

	tests := []struct {
		name string
		// oldRcByNode is the "from" deployment
		oldRcByNode *api.ReplicationController
		// newRcByNode is the "to" deployment
		newRcByNode *api.ReplicationController
		// whether newRcByNode existed (false means it was created)
		newRcByNodeExists bool
		nodeLabel         string
		// expected is the sequence of up/down events that will be simulated and
		// verified
		expected []interface{}
		// output is the expected textual output written
		output string
	}{
		{
			name:              "5->5",
			oldRcByNode:       oldRcByNode(5, 5),
			newRcByNode:       newRcByNode(0, 5),
			newRcByNodeExists: false,
			nodeLabel:         "node-label-foo",
			expected: []interface{}{
				down{oldReady: 5, newReady: 0, to: 4},
				down{oldReady: 4, newReady: 0, to: 3},
				down{oldReady: 3, newReady: 0, to: 2},
				up{2},
				down{oldReady: 2, newReady: 2, to: 1},
				down{oldReady: 1, newReady: 2, to: 0},
				up{5},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 5, scaling down foo-v1 from 5 to 0 (Node by Node)
Rolling update by node starting on node node1.
Scaling foo-v1 down to 4
Scaling foo-v1 down to 3
Scaling foo-v2 up to 0
Scaling foo-v2 up to 2
Rolling update by node finished on node node1.
Rolling update by node starting on node node2.
Scaling foo-v1 down to 2
Scaling foo-v1 down to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
Scaling foo-v2 up to 5
Rolling update by node finished on node node2.
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node1
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node2
`,
		},
		{
			name:              "5->7",
			oldRcByNode:       oldRcByNode(5, 5),
			newRcByNode:       newRcByNode(0, 7),
			newRcByNodeExists: false,
			nodeLabel:         "node-label-foo",
			expected: []interface{}{
				down{oldReady: 5, newReady: 0, to: 4},
				down{oldReady: 4, newReady: 0, to: 3},
				down{oldReady: 3, newReady: 0, to: 2},
				up{2},
				down{oldReady: 2, newReady: 2, to: 1},
				down{oldReady: 1, newReady: 2, to: 0},
				up{5},
				up{7},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 7, scaling down foo-v1 from 5 to 0 (Node by Node)
Rolling update by node starting on node node1.
Scaling foo-v1 down to 4
Scaling foo-v1 down to 3
Scaling foo-v2 up to 0
Scaling foo-v2 up to 2
Rolling update by node finished on node node1.
Rolling update by node starting on node node2.
Scaling foo-v1 down to 2
Scaling foo-v1 down to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
Scaling foo-v2 up to 5
Rolling update by node finished on node node2.
Final scaling Up Replication Controller foo-v2 to 7.
Scaling foo-v2 up to 7
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node1
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node2
`,
		},
		{
			name:              "5->4",
			oldRcByNode:       oldRcByNode(5, 5),
			newRcByNode:       newRcByNode(0, 4),
			newRcByNodeExists: false,
			nodeLabel:         "node-label-foo",
			expected: []interface{}{
				down{oldReady: 5, newReady: 0, to: 4},
				down{oldReady: 4, newReady: 0, to: 3},
				down{oldReady: 3, newReady: 0, to: 2},
				up{2},
				down{oldReady: 2, newReady: 2, to: 1},
				down{oldReady: 1, newReady: 2, to: 0},
				up{4},
			},
			output: `Created foo-v2
Scaling up foo-v2 from 0 to 4, scaling down foo-v1 from 5 to 0 (Node by Node)
Rolling update by node starting on node node1.
Scaling foo-v1 down to 4
Scaling foo-v1 down to 3
Scaling foo-v2 up to 0
Scaling foo-v2 up to 2
Rolling update by node finished on node node1.
Rolling update by node starting on node node2.
Scaling foo-v1 down to 2
Scaling foo-v1 down to 1
Scaling foo-v1 down to 0
Scaling foo-v2 up to 2
Scaling foo-v2 up to 4
Rolling update by node finished on node node2.
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node1
Delete annotation kubectl.kubernetes.io/desired-nodes--node-label-foo on node node2
`,
		},
	}

	for i, test := range tests {
		// Extract expectations into some makeshift FIFOs so they can be returned
		// in the correct order from the right places. This lets scale downs be
		// expressed a single event even though the data is used from multiple
		// interface calls.
		oldReady := []int{}
		newReady := []int{}
		upTo := []int{}
		downTo := []int{}
		for _, event := range test.expected {
			switch e := event.(type) {
			case down:
				oldReady = append(oldReady, e.oldReady)
				newReady = append(newReady, e.newReady)
				if !e.noop {
					downTo = append(downTo, e.to)
				}
			case up:
				upTo = append(upTo, e.to)
			}
		}

		// Make a way to get the next item from our FIFOs. Returns -1 if the array
		// is empty.
		next := func(s *[]int) int {
			slice := *s
			v := -1
			if len(slice) > 0 {
				v = slice[0]
				if len(slice) > 1 {
					*s = slice[1:]
				} else {
					*s = []int{}
				}
			}
			return v
		}
		t.Logf("running test %d (%s) (up: %v, down: %v, oldReady: %v, newReady: %v)", i, test.name, upTo, downTo, oldReady, newReady)
		OldRcByNode := oldRcByNode(5, 5)
		delete(OldRcByNode.Annotations, originalReplicasAnnotation)
		NewRcByNode := newRcByNode(0, 5)
		var updatedOldRc *api.ReplicationController
		node1 := Node("node1")
		node2 := Node("node2")
		nodeList := &api.NodeList{
			Items: []api.Node{*node1, *node2},
		}
		nodeMap := map[string]*api.Node{"node1": node1, "node2": node2}
		pod1 := Pod("pod1", "node1")
		pod2 := Pod("pod2", "node1")
		pod3 := Pod("pod3", "node2")
		pod4 := Pod("pod4", "node2")
		pod5 := Pod("pod5", "node2")
		podList := &api.PodList{
			Items: []api.Pod{*pod1, *pod2, *pod3, *pod4, *pod5},
		}
		node1podList := &api.PodList{
			Items: []api.Pod{*pod1, *pod2},
		}
		node2podList := &api.PodList{
			Items: []api.Pod{*pod3, *pod4, *pod5},
		}
		fake := &testclient.Fake{}
		fake.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
			switch a := action.(type) {
			case testclient.GetAction:
				getaction := action.(testclient.GetAction)
				if action.GetResource() == "nodes" {
					node, _ := nodeMap[getaction.GetName()]
					ret := []runtime.Object{node}
					return true, ret[0], nil
				} else {
					if getaction.GetName() == "foo-v1" {
						return true, OldRcByNode, nil
					} else if getaction.GetName() == "foo-v2" {
						return true, NewRcByNode, nil
					}
				}
			case testclient.UpdateAction:
				if action.GetResource() == "nodes" {
					updatedNode := a.GetObject().(*api.Node)
					if updatedNode.Name == "node1" {
						node1 = updatedNode
					} else if updatedNode.Name == "node2" {
						node2 = updatedNode
					}
					nodeList = &api.NodeList{
						Items: []api.Node{*node1, *node2},
					}
					return true, updatedNode, nil
				} else {
					updatedOldRc = a.GetObject().(*api.ReplicationController)
					return true, updatedOldRc, nil
				}
			case testclient.ListAction:
				if action.GetResource() == "nodes" {
					ret := []runtime.Object{nodeList}
					return true, ret[0], nil
				}
				if action.GetResource() == "pods" {
					listaction := action.(testclient.ListAction)
					ret := []runtime.Object{podList}
					if listaction.GetListRestrictions().Fields.String() == "spec.nodeName=node1" {
						ret = []runtime.Object{node1podList}
					} else if listaction.GetListRestrictions().Fields.String() == "spec.nodeName=node2" {
						ret = []runtime.Object{node2podList}
					}
					return true, ret[0], nil
				}
			}
			return false, nil, nil
		})

		updater := &RollingUpdaterByNode{
			c:  fake,
			ns: "default",
			scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
				// Return a scale up or scale down expectation depending on the rc,
				// and throw errors if there is no expectation expressed for this
				// call.
				expected := -1
				switch {
				case rc.Name == test.newRcByNode.Name:
					t.Logf("scaling up %s to %d", rc.Name, rc.Spec.Replicas)
					expected = next(&upTo)
				case rc.Name == test.oldRcByNode.Name:
					t.Logf("scaling down %s to %d", rc.Name, rc.Spec.Replicas)
					expected = next(&downTo)
				}
				if expected == -1 {
					t.Fatalf("unexpected scale of %s to %d", rc.Name, rc.Spec.Replicas)
				} else if e, a := expected, rc.Spec.Replicas; e != a {
					t.Fatalf("expected scale of %s to %d, got %d", rc.Name, e, a)
				}
				// Simulate the scale.
				rc.Status.Replicas = rc.Spec.Replicas
				return rc, nil
			},
			getOrCreateTargetController: func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
				// Simulate a create vs. update of an existing controller.
				return test.newRcByNode, test.newRcByNodeExists, nil
			},
			cleanup: func(oldRcByNode, newRcByNode *api.ReplicationController, config *RollingUpdaterByNodeConfig) error {
				return nil
			},
		}
		// Set up a mock readiness check which handles the test assertions.
		updater.getReadyPods = func(oldRcByNode, newRcByNode *api.ReplicationController) (int, int, error) {
			// Return simulated readiness, and throw an error if this call has no
			// expectations defined.
			oldReady := next(&oldReady)
			newReady := next(&newReady)
			if oldReady == -1 || newReady == -1 {
				t.Fatalf("unexpected getReadyPods call for:\noldRcByNode: %+v\nnewRcByNode: %+v", oldRcByNode, newRcByNode)
			}
			return oldReady, newReady, nil
		}
		var buffer bytes.Buffer
		config := &RollingUpdaterByNodeConfig{
			Out:           &buffer,
			OldRc:         test.oldRcByNode,
			NewRc:         test.newRcByNode,
			NodeLabel:     test.nodeLabel,
			UpdatePeriod:  0,
			Interval:      time.Millisecond,
			Timeout:       time.Millisecond,
			CleanupPolicy: DeleteRollingUpdateByNodeCleanupPolicy,
		}
		err := updater.Update(config)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if buffer.String() != test.output {
			t.Errorf("Bad output. expected:\n%s\ngot:\n%s", test.output, buffer.String())
		}
	}
}

func TestUpdateByNode_assignOriginalAnnotation(t *testing.T) {
	OldRcByNode := oldRcByNode(1, 1)
	delete(OldRcByNode.Annotations, originalReplicasAnnotation)
	NewRcByNode := newRcByNode(1, 1)
	var updatedOldRc *api.ReplicationController
	node1 := Node("node1")
	node2 := Node("node2")
	nodeList := &api.NodeList{
		Items: []api.Node{*node1, *node2},
	}
	nodeMap := map[string]*api.Node{"node1": node1, "node2": node2}

	pod1 := Pod("pod1", "node1")
	pod2 := Pod("pod2", "node1")
	pod3 := Pod("pod3", "node2")
	pod4 := Pod("pod4", "node2")
	pod5 := Pod("pod5", "node2")

	podList := &api.PodList{
		Items: []api.Pod{*pod1, *pod2, *pod3, *pod4, *pod5},
	}
	node1podList := &api.PodList{
		Items: []api.Pod{*pod1, *pod2},
	}
	node2podList := &api.PodList{
		Items: []api.Pod{*pod3, *pod4, *pod5},
	}

	fake := &testclient.Fake{}
	fake.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		switch a := action.(type) {
		case testclient.GetAction:
			getaction := action.(testclient.GetAction)
			if action.GetResource() == "nodes" {
				node, _ := nodeMap[getaction.GetName()]
				ret := []runtime.Object{node}
				return true, ret[0], nil
			} else {
				if getaction.GetName() == "foo-v1" {
					return true, OldRcByNode, nil
				} else if getaction.GetName() == "foo-v2" {
					return true, NewRcByNode, nil
				}
			}
		case testclient.UpdateAction:
			if action.GetResource() == "nodes" {
				updatedNode := a.GetObject().(*api.Node)
				if updatedNode.Name == "node1" {
					node1 = updatedNode
				} else if updatedNode.Name == "node2" {
					node2 = updatedNode
				}
				nodeList = &api.NodeList{
					Items: []api.Node{*node1, *node2},
				}
				return true, updatedNode, nil
			} else {
				updatedOldRc = a.GetObject().(*api.ReplicationController)
				return true, updatedOldRc, nil
			}
		case testclient.ListAction:
			if action.GetResource() == "nodes" {
				ret := []runtime.Object{nodeList}
				return true, ret[0], nil
			}
			if action.GetResource() == "pods" {
				listaction := action.(testclient.ListAction)
				ret := []runtime.Object{podList}
				if listaction.GetListRestrictions().Fields.String() == "spec.nodeName=node1" {
					ret = []runtime.Object{node1podList}
				} else if listaction.GetListRestrictions().Fields.String() == "spec.nodeName=node2" {
					ret = []runtime.Object{node2podList}
				}
				return true, ret[0], nil
			}
		}

		return false, nil, nil
	})
	updater := &RollingUpdaterByNode{
		c:  fake,
		ns: "default",
		scaleAndWait: func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
			return rc, nil
		},
		getOrCreateTargetController: func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
			return NewRcByNode, false, nil
		},
		cleanup: func(OldRcByNode, NewRcByNode *api.ReplicationController, config *RollingUpdaterByNodeConfig) error {
			return nil
		},
		getReadyPods: func(OldRcByNode, NewRcByNode *api.ReplicationController) (int, int, error) {
			return 1, 1, nil
		},
	}
	var buffer bytes.Buffer
	config := &RollingUpdaterByNodeConfig{
		Out:           &buffer,
		OldRc:         OldRcByNode,
		NewRc:         NewRcByNode,
		UpdatePeriod:  0,
		NodeLabel:     "node-label-foo",
		Interval:      time.Millisecond,
		Timeout:       time.Millisecond,
		CleanupPolicy: DeleteRollingUpdateByNodeCleanupPolicy,
	}
	err := updater.Update(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if updatedOldRc == nil {
		t.Fatalf("expected rc to be updated")
	}
	if e, a := "1", updatedOldRc.Annotations[originalReplicasAnnotation]; e != a {
		t.Fatalf("expected annotation value %s, got %s", e, a)
	}
}

func TestRollingUpdaterByNode_multipleContainersInPod(t *testing.T) {
	tests := []struct {
		oldRcByNode *api.ReplicationController
		newRcByNode *api.ReplicationController

		container     string
		image         string
		deploymentKey string
	}{
		{
			oldRcByNode: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "container1",
									Image: "image1",
								},
								{
									Name:  "container2",
									Image: "image2",
								},
							},
						},
					},
				},
			},
			newRcByNode: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "container1",
									Image: "newimage",
								},
								{
									Name:  "container2",
									Image: "image2",
								},
							},
						},
					},
				},
			},
			container:     "container1",
			image:         "newimage",
			deploymentKey: "dk",
		},
		{
			oldRcByNode: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "container1",
									Image: "image1",
								},
							},
						},
					},
				},
			},
			newRcByNode: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "old",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "old",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "container1",
									Image: "newimage",
								},
							},
						},
					},
				},
			},
			container:     "container1",
			image:         "newimage",
			deploymentKey: "dk",
		},
	}

	for _, test := range tests {
		fake := &testclient.Fake{}
		fake.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
			switch action.(type) {
			case testclient.GetAction:
				return true, test.oldRcByNode, nil
			}
			return false, nil, nil
		})

		codec := testapi.Default.Codec()

		deploymentHash, err := api.HashObject(test.newRcByNode, codec)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		test.newRcByNode.Spec.Selector[test.deploymentKey] = deploymentHash
		test.newRcByNode.Spec.Template.Labels[test.deploymentKey] = deploymentHash
		test.newRcByNode.Name = fmt.Sprintf("%s-%s", test.newRcByNode.Name, deploymentHash)

		updatedRc, err := CreateNewControllerFromCurrentController(fake, codec, "", test.oldRcByNode.ObjectMeta.Name, test.newRcByNode.ObjectMeta.Name, test.image, test.container, test.deploymentKey)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(updatedRc, test.newRcByNode) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.newRcByNode, updatedRc)
		}
	}
}

// TestRollingUpdaterByNode_cleanupWithClients ensures that the cleanup policy is
// correctly implemented.
func TestRollingUpdaterByNode_cleanupWithClients(t *testing.T) {
	rc := oldRcByNode(2, 2)
	rcExisting := newRcByNode(1, 3)

	tests := []struct {
		name      string
		policy    RollingUpdaterByNodeCleanupPolicy
		responses []runtime.Object
		expected  []string
	}{
		{
			name:      "preserve",
			policy:    PreserveRollingUpdateByNodeCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
			},
		},
		{
			name:      "delete",
			policy:    DeleteRollingUpdateByNodeCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
				"delete",
			},
		},
		{
			name:      "rename",
			policy:    RenameRollingUpdateByNodeCleanupPolicy,
			responses: []runtime.Object{rcExisting},
			expected: []string{
				"get",
				"update",
				"get",
				"get",
				"delete",
				"create",
				"delete",
			},
		},
	}

	for _, test := range tests {
		fake := testclient.NewSimpleFake(test.responses...)
		updater := &RollingUpdaterByNode{
			ns: "default",
			c:  fake,
		}
		config := &RollingUpdaterByNodeConfig{
			Out:           ioutil.Discard,
			OldRc:         rc,
			NewRc:         rcExisting,
			UpdatePeriod:  0,
			Interval:      time.Millisecond,
			Timeout:       time.Millisecond,
			CleanupPolicy: test.policy,
		}
		err := updater.cleanupWithClients(rc, rcExisting, config)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(fake.Actions()) != len(test.expected) {
			t.Fatalf("%s: unexpected actions: %v, expected %v", test.name, fake.Actions(), test.expected)
		}
		for j, action := range fake.Actions() {
			if e, a := test.expected[j], action.GetVerb(); e != a {
				t.Errorf("%s: unexpected action: expected %s, got %s", test.name, e, a)
			}
		}
	}
}

func TestUpdateByNodeExistingReplicationController(t *testing.T) {
	tests := []struct {
		rc              *api.ReplicationController
		name            string
		deploymentKey   string
		deploymentValue string

		expectedRc *api.ReplicationController
		expectErr  bool
	}{
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-hash",
							},
						},
					},
				},
			},
		},
		{
			rc: &api.ReplicationController{
				Spec: api.ReplicationControllerSpec{
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
				},
			},
			name:            "foo",
			deploymentKey:   "dk",
			deploymentValue: "some-hash",

			expectedRc: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubectl.kubernetes.io/next-controller-id": "foo",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{
						"dk": "some-other-hash",
					},
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"dk": "some-other-hash",
							},
						},
					},
				},
			},
		},
	}
	for _, test := range tests {
		buffer := &bytes.Buffer{}
		fakeClient := testclient.NewSimpleFake(test.expectedRc)
		rc, err := UpdateExistingReplicationController(fakeClient, test.rc, "default", test.name, test.deploymentKey, test.deploymentValue, buffer)
		if !reflect.DeepEqual(rc, test.expectedRc) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n", test.expectedRc, rc)
		}
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestUpdateByNodeWithRetries(t *testing.T) {
	codec := testapi.Default.Codec()
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "rc",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
	}
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
	}

	// Test end to end updating of the rc with retries. Essentially make sure the update handler
	// sees the right updates, failures in update/get are handled properly, and that the updated
	// rc with new resource version is returned to the caller. Without any of these rollingupdate
	// will fail cryptically.
	newRcByNode := *rc
	newRcByNode.ResourceVersion = "2"
	newRcByNode.Spec.Selector["baz"] = "foobar"
	updates := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, &newRcByNode)},
	}
	gets := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.ReplicationController{})},
		{StatusCode: 200, Body: objBody(codec, rc)},
	}
	nodes := []*http.Response{
		{StatusCode: 500, Body: objBody(codec, &api.Node{})},
		{StatusCode: 200, Body: objBody(codec, node)},
	}
	fakeClient := &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == testapi.Default.ResourcePath("replicationcontrollers", "default", "rc") && m == "PUT":
				update := updates[0]
				updates = updates[1:]
				// We should always get an update with a valid rc even when the get fails. The rc should always
				// contain the update.
				if c, ok := readOrDie(t, req, codec).(*api.ReplicationController); !ok || !reflect.DeepEqual(rc, c) {
					t.Errorf("Unexpected update body, got %+v expected %+v", c, rc)
				} else if sel, ok := c.Spec.Selector["baz"]; !ok || sel != "foobar" {
					t.Errorf("Expected selector label update, got %+v", c.Spec.Selector)
				} else {
					delete(c.Spec.Selector, "baz")
				}
				return update, nil
			case p == testapi.Default.ResourcePath("replicationcontrollers", "default", "rc") && m == "GET":
				get := gets[0]
				gets = gets[1:]
				return get, nil
			case p == testapi.Default.ResourcePath("nodes", api.NamespaceAll, "") && m == "GET":
				node := nodes[0]
				nodes = nodes[1:]
				return node, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	clientConfig := &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
	client := client.NewOrDie(clientConfig)
	client.Client = fakeClient.Client

	if rc, err := updateWithRetries(
		client.ReplicationControllers("default"), rc, func(c *api.ReplicationController) {
			c.Spec.Selector["baz"] = "foobar"
		}); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if sel, ok := rc.Spec.Selector["baz"]; !ok || sel != "foobar" || rc.ResourceVersion != "2" {
		t.Errorf("Expected updated rc, got %+v", rc)
	}
	if len(updates) != 0 || len(gets) != 0 {
		t.Errorf("Remaining updates %+v gets %+v", updates, gets)
	}
}

func TestRollingUpdaterByNode_readyPods(t *testing.T) {
	mkpod := func(owner *api.ReplicationController, ready bool) *api.Pod {
		labels := map[string]string{}
		for k, v := range owner.Spec.Selector {
			labels[k] = v
		}
		status := api.ConditionTrue
		if !ready {
			status = api.ConditionFalse
		}
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   "pod",
				Labels: labels,
			},
			Status: api.PodStatus{
				Conditions: []api.PodCondition{
					{
						Type:   api.PodReady,
						Status: status,
					},
				},
			},
		}
	}

	tests := []struct {
		oldRcByNode *api.ReplicationController
		newRcByNode *api.ReplicationController
		// expectated old/new ready counts
		oldReady int
		newReady int
		// pods owned by the rcs; indicate whether they're ready
		oldPods []bool
		newPods []bool
	}{
		{
			oldRcByNode: oldRcByNode(4, 4),
			newRcByNode: newRcByNode(4, 4),
			oldReady:    4,
			newReady:    2,
			oldPods: []bool{
				true,
				true,
				true,
				true,
			},
			newPods: []bool{
				true,
				false,
				true,
				false,
			},
		},
		{
			oldRcByNode: oldRcByNode(4, 4),
			newRcByNode: newRcByNode(4, 4),
			oldReady:    0,
			newReady:    1,
			oldPods: []bool{
				false,
			},
			newPods: []bool{
				true,
			},
		},
		{
			oldRcByNode: oldRcByNode(4, 4),
			newRcByNode: newRcByNode(4, 4),
			oldReady:    1,
			newReady:    0,
			oldPods: []bool{
				true,
			},
			newPods: []bool{
				false,
			},
		},
	}

	for i, test := range tests {
		t.Logf("evaluating test %d", i)
		// Populate the fake client with pods associated with their owners.
		pods := []runtime.Object{}
		for _, ready := range test.oldPods {
			pods = append(pods, mkpod(test.oldRcByNode, ready))
		}
		for _, ready := range test.newPods {
			pods = append(pods, mkpod(test.newRcByNode, ready))
		}
		client := testclient.NewSimpleFake(pods...)

		updater := &RollingUpdaterByNode{
			ns: "default",
			c:  client,
		}
		oldReady, newReady, err := updater.readyPods(test.oldRcByNode, test.newRcByNode)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if e, a := test.oldReady, oldReady; e != a {
			t.Errorf("expected old ready %d, got %d", e, a)
		}
		if e, a := test.newReady, newReady; e != a {
			t.Errorf("expected new ready %d, got %d", e, a)
		}
	}
}
