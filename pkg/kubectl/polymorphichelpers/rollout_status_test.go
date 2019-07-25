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

package polymorphichelpers

import (
	"fmt"
	"testing"

	apps "k8s.io/api/apps/v1"
	api "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubectl/pkg/scheme"
)

func TestDeploymentStatusViewerStatus(t *testing.T) {
	tests := []struct {
		name         string
		generation   int64
		specReplicas int32
		status       apps.DeploymentStatus
		msg          string
		done         bool
	}{
		{
			name:         "test1",
			generation:   0,
			specReplicas: 1,
			status: apps.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            1,
				UpdatedReplicas:     0,
				AvailableReplicas:   1,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for deployment \"foo\" rollout to finish: 0 out of 1 new replicas have been updated...\n",
			done: false,
		},
		{
			name:         "test2",
			generation:   1,
			specReplicas: 1,
			status: apps.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     1,
				AvailableReplicas:   2,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for deployment \"foo\" rollout to finish: 1 old replicas are pending termination...\n",
			done: false,
		},
		{
			name:         "test3",
			generation:   1,
			specReplicas: 2,
			status: apps.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   1,
				UnavailableReplicas: 1,
			},

			msg:  "Waiting for deployment \"foo\" rollout to finish: 1 of 2 updated replicas are available...\n",
			done: false,
		},
		{
			name:         "test4",
			generation:   1,
			specReplicas: 2,
			status: apps.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   2,
				UnavailableReplicas: 0,
			},

			msg:  "deployment \"foo\" successfully rolled out\n",
			done: true,
		},
		{
			name:         "test5",
			generation:   2,
			specReplicas: 2,
			status: apps.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   2,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for deployment spec update to be observed...\n",
			done: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			d := &apps.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:  "bar",
					Name:       "foo",
					UID:        "8764ae47-9092-11e4-8393-42010af018ff",
					Generation: test.generation,
				},
				Spec: apps.DeploymentSpec{
					Replicas: &test.specReplicas,
				},
				Status: test.status,
			}
			unstructuredD := &unstructured.Unstructured{}
			err := scheme.Scheme.Convert(d, unstructuredD, nil)
			if err != nil {
				t.Fatal(err)
			}

			dsv := &DeploymentStatusViewer{}
			msg, done, err := dsv.Status(unstructuredD, 0)
			if err != nil {
				t.Fatalf("DeploymentStatusViewer.Status(): %v", err)
			}
			if done != test.done || msg != test.msg {
				t.Errorf("DeploymentStatusViewer.Status() for deployment with generation %d, %d replicas specified, and status %+v returned %q, %t, want %q, %t",
					test.generation,
					test.specReplicas,
					test.status,
					msg,
					done,
					test.msg,
					test.done,
				)
			}
		})
	}
}

func TestDaemonSetStatusViewerStatus(t *testing.T) {
	tests := []struct {
		name       string
		generation int64
		status     apps.DaemonSetStatus
		msg        string
		done       bool
	}{
		{
			name:       "test1",
			generation: 0,
			status: apps.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 0,
				DesiredNumberScheduled: 1,
				NumberAvailable:        0,
			},

			msg:  "Waiting for daemon set \"foo\" rollout to finish: 0 out of 1 new pods have been updated...\n",
			done: false,
		},
		{
			name:       "test2",
			generation: 1,
			status: apps.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        1,
			},

			msg:  "Waiting for daemon set \"foo\" rollout to finish: 1 of 2 updated pods are available...\n",
			done: false,
		},
		{
			name:       "test3",
			generation: 1,
			status: apps.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        2,
			},

			msg:  "daemon set \"foo\" successfully rolled out\n",
			done: true,
		},
		{
			name:       "test4",
			generation: 2,
			status: apps.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        2,
			},

			msg:  "Waiting for daemon set spec update to be observed...\n",
			done: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			d := &apps.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:  "bar",
					Name:       "foo",
					UID:        "8764ae47-9092-11e4-8393-42010af018ff",
					Generation: test.generation,
				},
				Spec: apps.DaemonSetSpec{
					UpdateStrategy: apps.DaemonSetUpdateStrategy{
						Type: apps.RollingUpdateDaemonSetStrategyType,
					},
				},
				Status: test.status,
			}

			unstructuredD := &unstructured.Unstructured{}
			err := scheme.Scheme.Convert(d, unstructuredD, nil)
			if err != nil {
				t.Fatal(err)
			}

			dsv := &DaemonSetStatusViewer{}
			msg, done, err := dsv.Status(unstructuredD, 0)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if done != test.done || msg != test.msg {
				t.Errorf("daemon set with generation %d, %d pods specified, and status:\n%+v\nreturned:\n%q, %t\nwant:\n%q, %t",
					test.generation,
					d.Status.DesiredNumberScheduled,
					test.status,
					msg,
					done,
					test.msg,
					test.done,
				)
			}
		})
	}
}

func TestStatefulSetStatusViewerStatus(t *testing.T) {
	tests := []struct {
		name       string
		generation int64
		strategy   apps.StatefulSetUpdateStrategy
		status     apps.StatefulSetStatus
		msg        string
		done       bool
		err        bool
	}{
		{
			name:       "on delete returns an error",
			generation: 1,
			strategy:   apps.StatefulSetUpdateStrategy{Type: apps.OnDeleteStatefulSetStrategyType},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 1,
				Replicas:           0,
				ReadyReplicas:      1,
				CurrentReplicas:    0,
				UpdatedReplicas:    0,
			},

			msg:  "",
			done: true,
			err:  true,
		},
		{
			name:       "unobserved update is not complete",
			generation: 2,
			strategy:   apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 1,
				Replicas:           3,
				ReadyReplicas:      3,
				CurrentReplicas:    3,
				UpdatedReplicas:    0,
			},

			msg:  "Waiting for statefulset spec update to be observed...\n",
			done: false,
			err:  false,
		},
		{
			name:       "if all pods are not ready the update is not complete",
			generation: 1,
			strategy:   apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 2,
				Replicas:           3,
				ReadyReplicas:      2,
				CurrentReplicas:    3,
				UpdatedReplicas:    0,
			},

			msg:  fmt.Sprintf("Waiting for %d pods to be ready...\n", 1),
			done: false,
			err:  false,
		},
		{
			name:       "partition update completes when all replicas above the partition are updated",
			generation: 1,
			strategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
					partition := int32(2)
					return &apps.RollingUpdateStatefulSetStrategy{Partition: &partition}
				}()},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 2,
				Replicas:           3,
				ReadyReplicas:      3,
				CurrentReplicas:    2,
				UpdatedReplicas:    1,
			},

			msg:  fmt.Sprintf("partitioned roll out complete: %d new pods have been updated...\n", 1),
			done: true,
			err:  false,
		},
		{
			name:       "partition update is in progress if all pods above the partition have not been updated",
			generation: 1,
			strategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: func() *apps.RollingUpdateStatefulSetStrategy {
					partition := int32(2)
					return &apps.RollingUpdateStatefulSetStrategy{Partition: &partition}
				}()},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 2,
				Replicas:           3,
				ReadyReplicas:      3,
				CurrentReplicas:    3,
				UpdatedReplicas:    0,
			},

			msg:  fmt.Sprintf("Waiting for partitioned roll out to finish: %d out of %d new pods have been updated...\n", 0, 1),
			done: true,
			err:  false,
		},
		{
			name:       "update completes when all replicas are current",
			generation: 1,
			strategy:   apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			status: apps.StatefulSetStatus{
				ObservedGeneration: 2,
				Replicas:           3,
				ReadyReplicas:      3,
				CurrentReplicas:    3,
				UpdatedReplicas:    3,
				CurrentRevision:    "foo",
				UpdateRevision:     "foo",
			},

			msg:  fmt.Sprintf("statefulset rolling update complete %d pods at revision %s...\n", 3, "foo"),
			done: true,
			err:  false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := newStatefulSet(3)
			s.Status = test.status
			s.Spec.UpdateStrategy = test.strategy
			s.Generation = test.generation

			unstructuredS := &unstructured.Unstructured{}
			err := scheme.Scheme.Convert(s, unstructuredS, nil)
			if err != nil {
				t.Fatal(err)
			}

			dsv := &StatefulSetStatusViewer{}
			msg, done, err := dsv.Status(unstructuredS, 0)
			if test.err && err == nil {
				t.Fatalf("%s: expected error", test.name)
			}
			if !test.err && err != nil {
				t.Fatalf("%s: %s", test.name, err)
			}
			if done && !test.done {
				t.Errorf("%s: want done %v got %v", test.name, done, test.done)
			}
			if msg != test.msg {
				t.Errorf("%s: want message %s got %s", test.name, test.msg, msg)
			}
		})
	}
}

func TestDaemonSetStatusViewerStatusWithWrongUpdateStrategyType(t *testing.T) {
	d := &apps.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "bar",
			Name:      "foo",
			UID:       "8764ae47-9092-11e4-8393-42010af018ff",
		},
		Spec: apps.DaemonSetSpec{
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.OnDeleteDaemonSetStrategyType,
			},
		},
	}

	unstructuredD := &unstructured.Unstructured{}
	err := scheme.Scheme.Convert(d, unstructuredD, nil)
	if err != nil {
		t.Fatal(err)
	}

	dsv := &DaemonSetStatusViewer{}
	msg, done, err := dsv.Status(unstructuredD, 0)
	errMsg := "rollout status is only available for RollingUpdate strategy type"
	if err == nil || err.Error() != errMsg {
		t.Errorf("Status for daemon sets with UpdateStrategy type different than RollingUpdate should return error. Instead got: msg: %s\ndone: %t\n err: %v", msg, done, err)
	}
}

func newStatefulSet(replicas int32) *apps.StatefulSet {
	return &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector:            &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "test",
							Image:           "test_image",
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			Replicas:       &replicas,
			UpdateStrategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
		},
		Status: apps.StatefulSetStatus{},
	}
}
