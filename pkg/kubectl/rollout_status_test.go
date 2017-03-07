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

package kubectl

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	intstrutil "k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func intOrStringP(i int) *intstrutil.IntOrString {
	intstr := intstrutil.FromInt(i)
	return &intstr
}

func TestDeploymentStatusViewerStatus(t *testing.T) {
	tests := []struct {
		generation     int64
		specReplicas   int32
		maxUnavailable *intstrutil.IntOrString
		status         extensions.DeploymentStatus
		msg            string
		done           bool
	}{
		{
			generation:   0,
			specReplicas: 1,
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            1,
				UpdatedReplicas:     0,
				AvailableReplicas:   1,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for rollout to finish: 0 out of 1 new replicas have been updated...\n",
			done: false,
		},
		{
			generation:   1,
			specReplicas: 1,
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     1,
				AvailableReplicas:   2,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for rollout to finish: 1 old replicas are pending termination...\n",
			done: false,
		},
		{
			generation:     1,
			specReplicas:   2,
			maxUnavailable: intOrStringP(0),
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   1,
				UnavailableReplicas: 1,
			},

			msg:  "Waiting for rollout to finish: 1 of 2 updated replicas are available (minimum required: 2)...\n",
			done: false,
		},
		{
			generation:   1,
			specReplicas: 2,
			status: extensions.DeploymentStatus{
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
			generation:   2,
			specReplicas: 2,
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   2,
				UnavailableReplicas: 0,
			},

			msg:  "Waiting for deployment spec update to be observed...\n",
			done: false,
		},
		{
			generation:     1,
			specReplicas:   2,
			maxUnavailable: intOrStringP(1),
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   1,
				UnavailableReplicas: 0,
			},

			msg:  "deployment \"foo\" successfully rolled out\n",
			done: true,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Logf("testing scenario %d", i)
		d := &extensions.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:  "bar",
				Name:       "foo",
				UID:        "8764ae47-9092-11e4-8393-42010af018ff",
				Generation: test.generation,
			},
			Spec: extensions.DeploymentSpec{
				Strategy: extensions.DeploymentStrategy{
					Type: extensions.RollingUpdateDeploymentStrategyType,
					RollingUpdate: &extensions.RollingUpdateDeployment{
						MaxSurge: *intOrStringP(1),
					},
				},
				Replicas: test.specReplicas,
			},
			Status: test.status,
		}
		if test.maxUnavailable != nil {
			d.Spec.Strategy.RollingUpdate.MaxUnavailable = *test.maxUnavailable
		}
		client := fake.NewSimpleClientset(d).Extensions()
		dsv := &DeploymentStatusViewer{c: client}
		msg, done, err := dsv.Status("bar", "foo", 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if done != test.done || msg != test.msg {
			t.Errorf("deployment with generation %d, %d replicas specified, and status:\n%+v\nreturned:\n%q, %t\nwant:\n%q, %t",
				test.generation,
				test.specReplicas,
				test.status,
				msg,
				done,
				test.msg,
				test.done,
			)
		}
	}
}

func TestDaemonSetStatusViewerStatus(t *testing.T) {
	tests := []struct {
		generation     int64
		maxUnavailable *intstrutil.IntOrString
		status         extensions.DaemonSetStatus
		msg            string
		done           bool
	}{
		{
			generation: 0,
			status: extensions.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 0,
				DesiredNumberScheduled: 1,
				NumberAvailable:        0,
			},

			msg:  "Waiting for rollout to finish: 0 out of 1 new pods have been updated...\n",
			done: false,
		},
		{
			generation:     1,
			maxUnavailable: intOrStringP(0),
			status: extensions.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        1,
			},

			msg:  "Waiting for rollout to finish: 1 of 2 updated pods are available (minimum required: 2)...\n",
			done: false,
		},
		{
			generation: 1,
			status: extensions.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        2,
			},

			msg:  "daemon set \"foo\" successfully rolled out\n",
			done: true,
		},
		{
			generation: 2,
			status: extensions.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        2,
			},

			msg:  "Waiting for daemon set spec update to be observed...\n",
			done: false,
		},
		{
			generation:     1,
			maxUnavailable: intOrStringP(1),
			status: extensions.DaemonSetStatus{
				ObservedGeneration:     1,
				UpdatedNumberScheduled: 2,
				DesiredNumberScheduled: 2,
				NumberAvailable:        1,
			},

			msg:  "daemon set \"foo\" successfully rolled out\n",
			done: true,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Logf("testing scenario %d", i)
		d := &extensions.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:  "bar",
				Name:       "foo",
				UID:        "8764ae47-9092-11e4-8393-42010af018ff",
				Generation: test.generation,
			},
			Spec: extensions.DaemonSetSpec{
				UpdateStrategy: extensions.DaemonSetUpdateStrategy{
					Type:          extensions.RollingUpdateDaemonSetStrategyType,
					RollingUpdate: &extensions.RollingUpdateDaemonSet{},
				},
			},
			Status: test.status,
		}
		if test.maxUnavailable != nil {
			d.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable = *test.maxUnavailable
		}
		client := fake.NewSimpleClientset(d).Extensions()
		dsv := &DaemonSetStatusViewer{c: client}
		msg, done, err := dsv.Status("bar", "foo", 0)
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
	}
}

func TestDaemonSetStatusViewerStatusWithWrongUpdateStrategyType(t *testing.T) {
	d := &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "bar",
			Name:      "foo",
			UID:       "8764ae47-9092-11e4-8393-42010af018ff",
		},
		Spec: extensions.DaemonSetSpec{
			UpdateStrategy: extensions.DaemonSetUpdateStrategy{
				Type: extensions.OnDeleteDaemonSetStrategyType,
			},
		},
	}
	client := fake.NewSimpleClientset(d).Extensions()
	dsv := &DaemonSetStatusViewer{c: client}
	msg, done, err := dsv.Status("bar", "foo", 0)
	errMsg := "Status is available only for RollingUpdate strategy type"
	if err == nil || err.Error() != errMsg {
		t.Errorf("Status for daemon sets with UpdateStrategy type different than RollingUpdate should return error. Instead got: msg: %s\ndone: %t\n err: %v", msg, done, err)
	}
}
