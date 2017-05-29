/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	testclient "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

type fakeClientAccessFactory struct {
	ClientAccessFactory

	fakeClientset *fake.Clientset
}

func (f *fakeClientAccessFactory) ClientSetForVersion(requiredVersion *schema.GroupVersion) (internalclientset.Interface, error) {
	return f.fakeClientset, nil
}

func newFakeClientAccessFactory(objs []runtime.Object) *fakeClientAccessFactory {
	return &fakeClientAccessFactory{
		fakeClientset: fake.NewSimpleClientset(objs...),
	}
}

var (
	podsResource = schema.GroupVersionResource{Resource: "pods"}
	podsKind     = schema.GroupVersionKind{Kind: "Pod"}
)

func TestLogsForObject(t *testing.T) {
	tests := []struct {
		name    string
		obj     runtime.Object
		opts    *api.PodLogOptions
		pods    []runtime.Object
		actions []testclient.Action
	}{
		{
			name: "pod logs",
			obj: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				getLogsAction("test", nil),
			},
		},
		{
			name: "replication controller logs",
			obj: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: api.ReplicationControllerSpec{
					Selector: map[string]string{"foo": "bar"},
				},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				testclient.NewListAction(podsResource, podsKind, "test", metav1.ListOptions{LabelSelector: "foo=bar"}),
				getLogsAction("test", nil),
			},
		},
		{
			name: "replica set logs",
			obj: &extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: extensions.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				testclient.NewListAction(podsResource, podsKind, "test", metav1.ListOptions{LabelSelector: "foo=bar"}),
				getLogsAction("test", nil),
			},
		},
		{
			name: "deployment logs",
			obj: &extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: extensions.DeploymentSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				testclient.NewListAction(podsResource, podsKind, "test", metav1.ListOptions{LabelSelector: "foo=bar"}),
				getLogsAction("test", nil),
			},
		},
		{
			name: "job logs",
			obj: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				testclient.NewListAction(podsResource, podsKind, "test", metav1.ListOptions{LabelSelector: "foo=bar"}),
				getLogsAction("test", nil),
			},
		},
		{
			name: "stateful set logs",
			obj: &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: apps.StatefulSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			pods: []runtime.Object{testPod()},
			actions: []testclient.Action{
				testclient.NewListAction(podsResource, podsKind, "test", metav1.ListOptions{LabelSelector: "foo=bar"}),
				getLogsAction("test", nil),
			},
		},
	}

	for _, test := range tests {
		caf := newFakeClientAccessFactory(test.pods)
		omf := NewObjectMappingFactory(caf)
		_, err := omf.LogsForObject(test.obj, test.opts, 20*time.Second)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		}
		for i := range test.actions {
			if len(caf.fakeClientset.Actions()) < i {
				t.Errorf("%s: action %d does not exists in actual actions: %#v",
					test.name, i, caf.fakeClientset.Actions())
				continue
			}
			got := caf.fakeClientset.Actions()[i]
			want := test.actions[i]
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s: unexpected action: %s", test.name, diff.ObjectDiff(got, want))
			}
		}
	}
}

func testPod() runtime.Object {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "test",
			Labels:    map[string]string{"foo": "bar"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "c1"}},
		},
	}
}

func getLogsAction(namespace string, opts *api.PodLogOptions) testclient.Action {
	action := testclient.GenericActionImpl{}
	action.Verb = "get"
	action.Namespace = namespace
	action.Resource = podsResource
	action.Subresource = "logs"
	action.Value = opts
	return action
}
