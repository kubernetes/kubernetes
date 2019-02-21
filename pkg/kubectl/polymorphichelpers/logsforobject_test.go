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

package polymorphichelpers

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	fakeexternal "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
)

func TestLogsForObjectWithClient(t *testing.T) {
	namespace := "test"
	makeTestPod := func() runtime.Object {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: namespace,
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
			},
		}
	}

	getLogsReq := func(name string, opts *corev1.PodLogOptions) *rest.Request {
		return fakeexternal.NewSimpleClientset().CoreV1().Pods(namespace).GetLogs(name, opts)
	}

	tests := []struct {
		name             string
		obj              runtime.Object
		allContainers    bool
		store            []runtime.Object
		expectedRequests []*rest.Request
	}{
		{
			name: "pod logs",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: namespace},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("hello", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "pod logs: all containers",
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{Name: "initc1"},
						{Name: "initc2"},
					},
					Containers: []corev1.Container{
						{Name: "c1"},
						{Name: "c2"},
					},
				},
			},
			allContainers: true,
			store:         []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("hello", &corev1.PodLogOptions{Container: "initc1"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "initc2"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "c1"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "c2"}),
			},
		},
		{
			name: "pods list logs",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{
								{Name: "initc1"},
								{Name: "initc2"},
							},
							Containers: []corev1.Container{
								{Name: "c1"},
								{Name: "c2"},
							},
						},
					},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("hello", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "pods list logs: all containers",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{
								{Name: "initc1"},
								{Name: "initc2"},
							},
							Containers: []corev1.Container{
								{Name: "c1"},
								{Name: "c2"},
							},
						},
					},
				},
			},
			allContainers: true,
			store:         []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("hello", &corev1.PodLogOptions{Container: "initc1"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "initc2"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "c1"}),
				getLogsReq("hello", &corev1.PodLogOptions{Container: "c2"}),
			},
		},
		{
			name: "replication controller logs",
			obj: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: corev1.ReplicationControllerSpec{
					Selector: map[string]string{"foo": "bar"},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("foo", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "replica set logs",
			obj: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("foo", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "deployment logs",
			obj: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: extensionsv1beta1.DeploymentSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("foo", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "job logs",
			obj: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: batchv1.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("foo", &corev1.PodLogOptions{}),
			},
		},
		{
			name: "stateful set logs",
			obj: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: appsv1.StatefulSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			store: []runtime.Object{makeTestPod()},
			expectedRequests: []*rest.Request{
				getLogsReq("foo", &corev1.PodLogOptions{}),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeClientset := fakeexternal.NewSimpleClientset(tc.store...)
			requests, err := logsForObjectWithClient(fakeClientset.CoreV1(), tc.obj, &corev1.PodLogOptions{}, 20*time.Second, tc.allContainers)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(tc.expectedRequests, requests) {
				t.Logf("diff: %s", diff.ObjectReflectDiff(tc.expectedRequests, requests))
				t.Error(spew.Errorf("expected %#+v, got %#+v", tc.expectedRequests, requests))
			}
		})
	}
}
