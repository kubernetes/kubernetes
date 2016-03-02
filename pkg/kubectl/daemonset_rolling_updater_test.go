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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/watch"
)

func Node(name string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: nil,
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
func oldDs() *extensions.DaemonSet {
	return &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name: "foo-v1",
			UID:  "7764ae47-9092-11e4-8393-42010af018ff",
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels:      map[string]string{"version": "v1"},
				MatchExpressions: []unversioned.LabelSelectorRequirement{},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:   "foo-v1",
					Labels: map[string]string{"version": "v1"},
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
	}
}

func newDs() *extensions.DaemonSet {
	ds := oldDs()
	ds.Spec.Template = api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo-v2",
			Labels: map[string]string{"version": "v1"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "container11",
					Image: "image1",
				},
				{
					Name:  "container22",
					Image: "image2",
				},
			},
		},
	}
	ds.Spec.Selector = &unversioned.LabelSelector{
		MatchLabels:      map[string]string{"version": "v1"},
		MatchExpressions: []unversioned.LabelSelectorRequirement{},
	}
	ds.ObjectMeta = api.ObjectMeta{
		Name: "foo-v2",
	}
	return ds
}

// TestUpdate performs complex scenario testing for rolling updates. It
// provides fine grained control over the states for each update interval to
// allow the expression of as many edge cases as possible.
func TestDaemonSetUpdate(t *testing.T) {

	pod1 := Pod("pod1", "node1")
	pod1.Status = api.PodStatus{
		Conditions: []api.PodCondition{
			{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
	}
	pod2 := Pod("pod2", "node2")
	pod2.Status = api.PodStatus{
		Conditions: []api.PodCondition{
			{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
	}

	podList := &api.PodList{
		Items: []api.Pod{*pod1, *pod2},
	}
	node1podList := &api.PodList{
		Items: []api.Pod{*pod1},
	}
	node2podList := &api.PodList{
		Items: []api.Pod{*pod2},
	}

	watchInterface := watch.NewFake()

	event_generator := func() {
		watchInterface.Delete(pod1)
		watchInterface.Add(pod1)
		watchInterface.Delete(pod2)
		watchInterface.Add(pod2)
		watchInterface.Stop()
	}

	tests := []struct {
		name string
		// oldDs is the "from" deployment
		oldDs *extensions.DaemonSet
		// newDs is the "to" deployment
		newDs *extensions.DaemonSet
		// whether newDs existed (false means it was created)
		newDsExists bool
		maxUnavail  intstr.IntOrString
		maxSurge    intstr.IntOrString
		// expected is the sequence of up/down events that will be simulated and
		// verified
		expected []interface{}
		// output is the expected textual output written
		output string
	}{
		{
			name:     "test update",
			oldDs:    oldDs(),
			newDs:    newDs(),
			expected: []interface{}{},
			output: `Created foo-v2
Deleted foo-v1
`,
		},
	}

	for _, test := range tests {
		// Extract expectations into some makeshift FIFOs so they can be returned
		// in the correct order from the right places. This lets scale downs be
		// expressed a single event even though the data is used from multiple
		// interface calls.
		var updatedoldDs *extensions.DaemonSet
		fakec := &testclient.Fake{}
		fakec.AddReactor("*", "*", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
			switch a := action.(type) {
			case testclient.GetAction:
				if action.GetResource() == "pods" {
					getaction := action.(testclient.GetAction)
					if getaction.GetName() == "pod1" {
						return true, pod1, nil
					} else if getaction.GetName() == "pod2" {
						return true, pod2, nil
					}
				} else {
					return true, test.oldDs, nil
				}
			case testclient.UpdateAction:
				updatedoldDs = a.GetObject().(*extensions.DaemonSet)
				return true, updatedoldDs, nil
			case testclient.ListAction:
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

		// Return always the same watcher so that we can feed it with event_generator
		fakec.AddWatchReactor("*", func(action testclient.Action) (bool, watch.Interface, error) {
			return true, watchInterface, nil
		})

		updater := &DaemonSetRollingUpdater{
			ns: "default",
			c:  fakec,
		}

		var buffer bytes.Buffer
		config := &DaemonSetRollingUpdaterConfig{
			Out:       &buffer,
			OldDs:     test.oldDs,
			NewDs:     test.newDs,
			RInterval: time.Millisecond,
			DInterval: time.Millisecond,
		}

		// Send expected event fo update
		fmt.Println("Starting event generator")
		go event_generator()
		// Do update
		fmt.Println("Starting update")
		err := updater.Update(config)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if buffer.String() != test.output {
			t.Errorf("Bad output. expected:\n%s\ngot:\n%s", test.output, buffer.String())
		}
	}
}
