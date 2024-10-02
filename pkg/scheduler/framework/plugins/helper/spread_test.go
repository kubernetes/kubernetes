/*
Copyright 2020 The Kubernetes Authors.

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

package helper

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestGetPodServices(t *testing.T) {
	fakeInformerFactory := informers.NewSharedInformerFactory(&fake.Clientset{}, 0*time.Second)
	var services []*v1.Service
	for i := 0; i < 3; i++ {
		service := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("service-%d", i),
				Namespace: "test",
			},
			Spec: v1.ServiceSpec{
				Selector: map[string]string{
					"app": fmt.Sprintf("test-%d", i),
				},
			},
		}
		services = append(services, service)
		fakeInformerFactory.Core().V1().Services().Informer().GetStore().Add(service)
	}
	var pods []*v1.Pod
	for i := 0; i < 5; i++ {
		pod := st.MakePod().Name(fmt.Sprintf("test-pod-%d", i)).
			Namespace("test").
			Label("app", fmt.Sprintf("test-%d", i)).
			Label("label", fmt.Sprintf("label-%d", i)).
			Obj()
		pods = append(pods, pod)
	}

	tests := []struct {
		name   string
		pod    *v1.Pod
		expect []*v1.Service
	}{
		{
			name:   "GetPodServices for pod-0",
			pod:    pods[0],
			expect: []*v1.Service{services[0]},
		},
		{
			name:   "GetPodServices for pod-1",
			pod:    pods[1],
			expect: []*v1.Service{services[1]},
		},
		{
			name:   "GetPodServices for pod-2",
			pod:    pods[2],
			expect: []*v1.Service{services[2]},
		},
		{
			name:   "GetPodServices for pod-3",
			pod:    pods[3],
			expect: nil,
		},
		{
			name:   "GetPodServices for pod-4",
			pod:    pods[4],
			expect: nil,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			get, err := GetPodServices(fakeInformerFactory.Core().V1().Services().Lister(), test.pod)
			if err != nil {
				t.Errorf("Error from GetPodServices: %v", err)
			} else if diff := cmp.Diff(test.expect, get); diff != "" {
				t.Errorf("Unexpected services (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestDefaultSelector(t *testing.T) {
	const (
		namespace                 = "test"
		podName                   = "test-pod"
		serviceName               = "test-service"
		replicaSetName            = "test-replicaset"
		replicationControllerName = "test-replicationcontroller"
		statefulSetName           = "test-statefulset"

		podLabelKey = "podLabelKey"
		podLabelVal = "podLabelVal"

		replicaSetLabelKey = "replicaSetLabelKey"
		replicaSetLabelVal = "replicaSetLabelVal"

		replicationLabelKey = "replicationLabelKey"
		replicationLabelVal = "replicationLabelVal"

		statefulSetLabelKey = "statefulSetLabelKey"
		statefulSetLabelVal = "statefulSetLabelVal"
	)

	fakeInformerFactory := informers.NewSharedInformerFactory(&fake.Clientset{}, 0*time.Second)

	// Create fake service
	addFakeService := func() error {
		// Create fake service
		service := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: namespace,
			},
			Spec: v1.ServiceSpec{
				Selector: map[string]string{
					podLabelKey: podLabelVal,
				},
			},
		}
		return fakeInformerFactory.Core().V1().Services().Informer().GetStore().Add(service)
	}

	// Create fake ReplicaSet
	addFakeReplicaSet := func() error {
		replicaSet := &appsv1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:      replicaSetName,
				Namespace: namespace,
			},
			Spec: appsv1.ReplicaSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						replicaSetLabelKey: replicaSetLabelVal,
					},
				},
			},
		}
		return fakeInformerFactory.Apps().V1().ReplicaSets().Informer().GetStore().Add(replicaSet)
	}

	// Create fake ReplicationController
	addFakeReplicationController := func() error {
		replicationController := &v1.ReplicationController{
			ObjectMeta: metav1.ObjectMeta{
				Name:      replicationControllerName,
				Namespace: namespace,
			},
			Spec: v1.ReplicationControllerSpec{
				Selector: map[string]string{
					replicationLabelKey: replicationLabelVal,
				},
			},
		}
		return fakeInformerFactory.Core().V1().ReplicationControllers().Informer().GetStore().Add(replicationController)
	}

	// Create fake StatefulSet
	addFakeStatefulSet := func() error {
		statefulSet := &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:      statefulSetName,
				Namespace: namespace,
			},
			Spec: appsv1.StatefulSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						statefulSetLabelKey: statefulSetLabelVal,
					},
				},
			},
		}
		return fakeInformerFactory.Apps().V1().StatefulSets().Informer().GetStore().Add(statefulSet)
	}

	tests := []struct {
		name                string
		pod                 *v1.Pod
		expect              labels.Set
		addFakeResourceList []func() error
	}{
		{
			name: "DefaultSelector for default case",
			pod: st.MakePod().Name(podName).
				Namespace(namespace).Label(podLabelKey, podLabelVal).
				Obj(),
			expect:              labels.Set{},
			addFakeResourceList: nil,
		},
		{
			name: "DefaultSelector for no OwnerReference pod case",
			pod: st.MakePod().Name(podName).
				Namespace(namespace).Label(podLabelKey, podLabelVal).
				Obj(),
			expect:              labels.Set{podLabelKey: podLabelVal},
			addFakeResourceList: []func() error{addFakeService},
		},
		{
			name: "DefaultSelector for ReplicaSet OwnerReference pod case",
			pod: st.MakePod().Name(podName).
				Namespace(namespace).Label(podLabelKey, podLabelVal).
				OwnerReference(replicaSetName, appsv1.SchemeGroupVersion.WithKind("ReplicaSet")).Obj(),
			expect:              labels.Set{podLabelKey: podLabelVal, replicaSetLabelKey: replicaSetLabelVal},
			addFakeResourceList: []func() error{addFakeService, addFakeReplicaSet},
		},
		{
			name: "DefaultSelector for ReplicationController OwnerReference pod case",
			pod: st.MakePod().Name(podName).
				Namespace(namespace).Label(podLabelKey, podLabelVal).
				OwnerReference(replicationControllerName, v1.SchemeGroupVersion.WithKind("ReplicationController")).Obj(),
			expect:              labels.Set{podLabelKey: podLabelVal, replicationLabelKey: replicationLabelVal},
			addFakeResourceList: []func() error{addFakeService, addFakeReplicationController},
		},
		{
			name: "DefaultSelector for StatefulSet OwnerReference pod case",
			pod: st.MakePod().Name(podName).
				Namespace(namespace).Label(podLabelKey, podLabelVal).
				OwnerReference(statefulSetName, appsv1.SchemeGroupVersion.WithKind("StatefulSet")).Obj(),
			expect:              labels.Set{podLabelKey: podLabelVal, statefulSetLabelKey: statefulSetLabelVal},
			addFakeResourceList: []func() error{addFakeService, addFakeStatefulSet},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Add fake resources if needed.
			if test.addFakeResourceList != nil {
				for _, addFakeResource := range test.addFakeResourceList {
					err := addFakeResource()
					if err != nil {
						t.Fatalf("failed to add fake resource: %v", err)
					}
				}
			}

			get := DefaultSelector(test.pod,
				fakeInformerFactory.Core().V1().Services().Lister(),
				fakeInformerFactory.Core().V1().ReplicationControllers().Lister(),
				fakeInformerFactory.Apps().V1().ReplicaSets().Lister(),
				fakeInformerFactory.Apps().V1().StatefulSets().Lister())
			diff := cmp.Diff(test.expect.String(), get.String())
			if diff != "" {
				t.Errorf("Unexpected services (-want, +got):\n%s", diff)
			}

		})
	}
}
