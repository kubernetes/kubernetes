/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/conversion"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	EvictionMethod = "Eviction"
	DeleteMethod   = "Delete"
)

var node *api.Node
var cordoned_node *api.Node

func TestMain(m *testing.M) {
	// Create a node.
	node = &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:              "node",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
		},
		Spec: api.NodeSpec{
			ExternalID: "node",
		},
		Status: api.NodeStatus{},
	}
	clone, _ := conversion.NewCloner().DeepCopy(node)

	// A copy of the same node, but cordoned.
	cordoned_node = clone.(*api.Node)
	cordoned_node.Spec.Unschedulable = true
	os.Exit(m.Run())
}

func TestCordon(t *testing.T) {
	tests := []struct {
		description string
		node        *api.Node
		expected    *api.Node
		cmd         func(cmdutil.Factory, io.Writer) *cobra.Command
		arg         string
		expectFatal bool
	}{
		{
			description: "node/node syntax",
			node:        cordoned_node,
			expected:    node,
			cmd:         NewCmdUncordon,
			arg:         "node/node",
			expectFatal: false,
		},
		{
			description: "uncordon for real",
			node:        cordoned_node,
			expected:    node,
			cmd:         NewCmdUncordon,
			arg:         "node",
			expectFatal: false,
		},
		{
			description: "uncordon does nothing",
			node:        node,
			expected:    node,
			cmd:         NewCmdUncordon,
			arg:         "node",
			expectFatal: false,
		},
		{
			description: "cordon does nothing",
			node:        cordoned_node,
			expected:    cordoned_node,
			cmd:         NewCmdCordon,
			arg:         "node",
			expectFatal: false,
		},
		{
			description: "cordon for real",
			node:        node,
			expected:    cordoned_node,
			cmd:         NewCmdCordon,
			arg:         "node",
			expectFatal: false,
		},
		{
			description: "cordon missing node",
			node:        node,
			expected:    node,
			cmd:         NewCmdCordon,
			arg:         "bar",
			expectFatal: true,
		},
		{
			description: "uncordon missing node",
			node:        node,
			expected:    node,
			cmd:         NewCmdUncordon,
			arg:         "bar",
			expectFatal: true,
		},
	}

	for _, test := range tests {
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		new_node := &api.Node{}
		updated := false
		tf.Client = &fake.RESTClient{
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				m := &MyReq{req}
				switch {
				case m.isFor("GET", "/nodes/node"):
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, test.node)}, nil
				case m.isFor("GET", "/nodes/bar"):
					return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: stringBody("nope")}, nil
				case m.isFor("PUT", "/nodes/node"):
					data, err := ioutil.ReadAll(req.Body)
					if err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					defer req.Body.Close()
					if err := runtime.DecodeInto(codec, data, new_node); err != nil {
						t.Fatalf("%s: unexpected error: %v", test.description, err)
					}
					if !reflect.DeepEqual(test.expected.Spec, new_node.Spec) {
						t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, test.expected.Spec, new_node.Spec)
					}
					updated = true
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, new_node)}, nil
				default:
					t.Fatalf("%s: unexpected request: %v %#v\n%#v", test.description, req.Method, req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.ClientConfig = defaultClientConfig()

		buf := bytes.NewBuffer([]byte{})
		cmd := test.cmd(f, buf)

		saw_fatal := false
		func() {
			defer func() {
				// Recover from the panic below.
				_ = recover()
				// Restore cmdutil behavior
				cmdutil.DefaultBehaviorOnFatal()
			}()
			cmdutil.BehaviorOnFatal(func(e string, code int) {
				saw_fatal = true
				panic(e)
			})
			cmd.SetArgs([]string{test.arg})
			cmd.Execute()
		}()

		if test.expectFatal {
			if !saw_fatal {
				t.Fatalf("%s: unexpected non-error", test.description)
			}
			if updated {
				t.Fatalf("%s: unexpcted update", test.description)
			}
		}

		if !test.expectFatal && saw_fatal {
			t.Fatalf("%s: unexpected error", test.description)
		}
		if !reflect.DeepEqual(test.expected.Spec, test.node.Spec) && !updated {
			t.Fatalf("%s: node never updated", test.description)
		}
	}
}

func TestDrain(t *testing.T) {
	labels := make(map[string]string)
	labels["my_key"] = "my_value"

	rc := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:              "rc",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			SelfLink:          testapi.Default.SelfLink("replicationcontrollers", "rc"),
		},
		Spec: api.ReplicationControllerSpec{
			Selector: labels,
		},
	}

	rc_anno := make(map[string]string)
	rc_anno[api.CreatedByAnnotation] = refJson(t, &rc)

	rc_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			Annotations:       rc_anno,
		},
		Spec: api.PodSpec{
			NodeName: "node",
		},
	}

	ds := extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name:              "ds",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			SelfLink:          "/apis/extensions/v1beta1/namespaces/default/daemonsets/ds",
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: labels},
		},
	}

	ds_anno := make(map[string]string)
	ds_anno[api.CreatedByAnnotation] = refJson(t, &ds)

	ds_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			Annotations:       ds_anno,
		},
		Spec: api.PodSpec{
			NodeName: "node",
		},
	}

	job := batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:              "job",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			SelfLink:          "/apis/extensions/v1beta1/namespaces/default/jobs/job",
		},
		Spec: batch.JobSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: labels},
		},
	}

	job_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			Annotations:       map[string]string{api.CreatedByAnnotation: refJson(t, &job)},
		},
	}

	rs := extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:              "rs",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			SelfLink:          testapi.Default.SelfLink("replicasets", "rs"),
		},
		Spec: extensions.ReplicaSetSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: labels},
		},
	}

	rs_anno := make(map[string]string)
	rs_anno[api.CreatedByAnnotation] = refJson(t, &rs)

	rs_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
			Annotations:       rs_anno,
		},
		Spec: api.PodSpec{
			NodeName: "node",
		},
	}

	naked_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
		},
		Spec: api.PodSpec{
			NodeName: "node",
		},
	}

	emptydir_pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "bar",
			Namespace:         "default",
			CreationTimestamp: unversioned.Time{Time: time.Now()},
			Labels:            labels,
		},
		Spec: api.PodSpec{
			NodeName: "node",
			Volumes: []api.Volume{
				{
					Name:         "scratch",
					VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: ""}},
				},
			},
		},
	}

	tests := []struct {
		description  string
		node         *api.Node
		expected     *api.Node
		pods         []api.Pod
		rcs          []api.ReplicationController
		replicaSets  []extensions.ReplicaSet
		args         []string
		expectFatal  bool
		expectDelete bool
	}{
		{
			description:  "RC-managed pod",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{rc_pod},
			rcs:          []api.ReplicationController{rc},
			args:         []string{"node"},
			expectFatal:  false,
			expectDelete: true,
		},
		{
			description:  "DS-managed pod",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{ds_pod},
			rcs:          []api.ReplicationController{rc},
			args:         []string{"node"},
			expectFatal:  true,
			expectDelete: false,
		},
		{
			description:  "DS-managed pod with --ignore-daemonsets",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{ds_pod},
			rcs:          []api.ReplicationController{rc},
			args:         []string{"node", "--ignore-daemonsets"},
			expectFatal:  false,
			expectDelete: false,
		},
		{
			description:  "Job-managed pod",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{job_pod},
			rcs:          []api.ReplicationController{rc},
			args:         []string{"node"},
			expectFatal:  false,
			expectDelete: true,
		},
		{
			description:  "RS-managed pod",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{rs_pod},
			replicaSets:  []extensions.ReplicaSet{rs},
			args:         []string{"node"},
			expectFatal:  false,
			expectDelete: true,
		},
		{
			description:  "naked pod",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{naked_pod},
			rcs:          []api.ReplicationController{},
			args:         []string{"node"},
			expectFatal:  true,
			expectDelete: false,
		},
		{
			description:  "naked pod with --force",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{naked_pod},
			rcs:          []api.ReplicationController{},
			args:         []string{"node", "--force"},
			expectFatal:  false,
			expectDelete: true,
		},
		{
			description:  "pod with EmptyDir",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{emptydir_pod},
			args:         []string{"node", "--force"},
			expectFatal:  true,
			expectDelete: false,
		},
		{
			description:  "pod with EmptyDir and --delete-local-data",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{emptydir_pod},
			args:         []string{"node", "--force", "--delete-local-data=true"},
			expectFatal:  false,
			expectDelete: true,
		},
		{
			description:  "empty node",
			node:         node,
			expected:     cordoned_node,
			pods:         []api.Pod{},
			rcs:          []api.ReplicationController{rc},
			args:         []string{"node"},
			expectFatal:  false,
			expectDelete: false,
		},
	}

	testEviction := false
	for i := 0; i < 2; i++ {
		testEviction = !testEviction
		var currMethod string
		if testEviction {
			currMethod = EvictionMethod
		} else {
			currMethod = DeleteMethod
		}
		for _, test := range tests {
			new_node := &api.Node{}
			deleted := false
			evicted := false
			f, tf, codec, ns := cmdtesting.NewAPIFactory()
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					m := &MyReq{req}
					switch {
					case req.Method == "GET" && req.URL.Path == "/api":
						apiVersions := unversioned.APIVersions{
							Versions: []string{"v1"},
						}
						return genResponseWithJsonEncodedBody(apiVersions)
					case req.Method == "GET" && req.URL.Path == "/apis":
						groupList := unversioned.APIGroupList{
							Groups: []unversioned.APIGroup{
								{
									Name: "policy",
									PreferredVersion: unversioned.GroupVersionForDiscovery{
										GroupVersion: "policy/v1beta1",
									},
								},
							},
						}
						return genResponseWithJsonEncodedBody(groupList)
					case req.Method == "GET" && req.URL.Path == "/api/v1":
						resourceList := unversioned.APIResourceList{
							GroupVersion: "v1",
						}
						if testEviction {
							resourceList.APIResources = []unversioned.APIResource{
								{
									Name: EvictionSubresource,
									Kind: EvictionKind,
								},
							}
						}
						return genResponseWithJsonEncodedBody(resourceList)
					case m.isFor("GET", "/nodes/node"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, test.node)}, nil
					case m.isFor("GET", "/namespaces/default/replicationcontrollers/rc"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &test.rcs[0])}, nil
					case m.isFor("GET", "/namespaces/default/daemonsets/ds"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(testapi.Extensions.Codec(), &ds)}, nil
					case m.isFor("GET", "/namespaces/default/jobs/job"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(testapi.Extensions.Codec(), &job)}, nil
					case m.isFor("GET", "/namespaces/default/replicasets/rs"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(testapi.Extensions.Codec(), &test.replicaSets[0])}, nil
					case m.isFor("GET", "/namespaces/default/pods/bar"):
						return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &api.Pod{})}, nil
					case m.isFor("GET", "/pods"):
						values, err := url.ParseQuery(req.URL.RawQuery)
						if err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						get_params := make(url.Values)
						get_params["fieldSelector"] = []string{"spec.nodeName=node"}
						if !reflect.DeepEqual(get_params, values) {
							t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, get_params, values)
						}
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.PodList{Items: test.pods})}, nil
					case m.isFor("GET", "/replicationcontrollers"):
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.ReplicationControllerList{Items: test.rcs})}, nil
					case m.isFor("PUT", "/nodes/node"):
						data, err := ioutil.ReadAll(req.Body)
						if err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						defer req.Body.Close()
						if err := runtime.DecodeInto(codec, data, new_node); err != nil {
							t.Fatalf("%s: unexpected error: %v", test.description, err)
						}
						if !reflect.DeepEqual(test.expected.Spec, new_node.Spec) {
							t.Fatalf("%s: expected:\n%v\nsaw:\n%v\n", test.description, test.expected.Spec, new_node.Spec)
						}
						return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, new_node)}, nil
					case m.isFor("DELETE", "/namespaces/default/pods/bar"):
						deleted = true
						return &http.Response{StatusCode: 204, Header: defaultHeader(), Body: objBody(codec, &test.pods[0])}, nil
					case m.isFor("POST", "/namespaces/default/pods/bar/eviction"):
						evicted = true
						return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: policyObjBody(&policy.Eviction{})}, nil
					default:
						t.Fatalf("%s: unexpected request: %v %#v\n%#v", test.description, req.Method, req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfig = defaultClientConfig()

			buf := bytes.NewBuffer([]byte{})
			errBuf := bytes.NewBuffer([]byte{})
			cmd := NewCmdDrain(f, buf, errBuf)

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
					t.Fatalf("%s: unexpected non-error when using %s", test.description, currMethod)
				}
			}

			if test.expectDelete {
				// Test Delete
				if !testEviction && !deleted {
					t.Fatalf("%s: pod never deleted", test.description)
				}
				// Test Eviction
				if testEviction && !evicted {
					t.Fatalf("%s: pod never evicted", test.description)
				}
			}
			if !test.expectDelete {
				if deleted {
					t.Fatalf("%s: unexpected delete when using %s", test.description, currMethod)
				}
			}
		}
	}
}

func TestDeletePods(t *testing.T) {
	ifHasBeenCalled := map[string]bool{}
	tests := []struct {
		description       string
		interval          time.Duration
		timeout           time.Duration
		expectPendingPods bool
		expectError       bool
		expectedError     *error
		getPodFn          func(namespace, name string) (*api.Pod, error)
	}{
		{
			description:       "Wait for deleting to complete",
			interval:          100 * time.Millisecond,
			timeout:           10 * time.Second,
			expectPendingPods: false,
			expectError:       false,
			expectedError:     nil,
			getPodFn: func(namespace, name string) (*api.Pod, error) {
				oldPodMap, _ := createPods(false)
				newPodMap, _ := createPods(true)
				if oldPod, found := oldPodMap[name]; found {
					if _, ok := ifHasBeenCalled[name]; !ok {
						ifHasBeenCalled[name] = true
						return &oldPod, nil
					} else {
						if oldPod.ObjectMeta.Generation < 4 {
							newPod := newPodMap[name]
							return &newPod, nil
						} else {
							return nil, apierrors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, name)
						}
					}
				}
				return nil, apierrors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, name)
			},
		},
		{
			description:       "Deleting could timeout",
			interval:          200 * time.Millisecond,
			timeout:           3 * time.Second,
			expectPendingPods: true,
			expectError:       true,
			expectedError:     &wait.ErrWaitTimeout,
			getPodFn: func(namespace, name string) (*api.Pod, error) {
				oldPodMap, _ := createPods(false)
				if oldPod, found := oldPodMap[name]; found {
					return &oldPod, nil
				}
				return nil, errors.New(fmt.Sprintf("%q: not found", name))
			},
		},
		{
			description:       "Client error could be passed out",
			interval:          200 * time.Millisecond,
			timeout:           5 * time.Second,
			expectPendingPods: true,
			expectError:       true,
			expectedError:     nil,
			getPodFn: func(namespace, name string) (*api.Pod, error) {
				return nil, errors.New("This is a random error for testing")
			},
		},
	}

	for _, test := range tests {
		f, _, _, _ := cmdtesting.NewAPIFactory()
		o := DrainOptions{}
		o.mapper, _ = f.Object()
		o.out = os.Stdout
		_, pods := createPods(false)
		pendingPods, err := o.waitForDelete(pods, test.interval, test.timeout, false, test.getPodFn)

		if test.expectError {
			if err == nil {
				t.Fatalf("%s: unexpected non-error", test.description)
			} else if test.expectedError != nil {
				if *test.expectedError != err {
					t.Fatalf("%s: the error does not match expected error", test.description)
				}
			}
		}
		if !test.expectError && err != nil {
			t.Fatalf("%s: unexpected error", test.description)
		}
		if test.expectPendingPods && len(pendingPods) == 0 {
			t.Fatalf("%s: unexpected empty pods", test.description)
		}
		if !test.expectPendingPods && len(pendingPods) > 0 {
			t.Fatalf("%s: unexpected pending pods", test.description)
		}
	}
}

func createPods(ifCreateNewPods bool) (map[string]api.Pod, []api.Pod) {
	podMap := make(map[string]api.Pod)
	podSlice := []api.Pod{}
	for i := 0; i < 8; i++ {
		var uid types.UID
		if ifCreateNewPods {
			uid = types.UID(i)
		} else {
			uid = types.UID(strconv.Itoa(i) + strconv.Itoa(i))
		}
		pod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:       "pod" + strconv.Itoa(i),
				Namespace:  "default",
				UID:        uid,
				Generation: int64(i),
			},
		}
		podMap[pod.Name] = pod
		podSlice = append(podSlice, pod)
	}
	return podMap, podSlice
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

func refJson(t *testing.T, o runtime.Object) string {
	ref, err := api.GetReference(o)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, _, codec, _ := cmdtesting.NewAPIFactory()
	json, err := runtime.Encode(codec, &api.SerializedReference{Reference: *ref})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	return string(json)
}
