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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestDeploymentStatusViewerStatus(t *testing.T) {
	tests := []struct {
		generation   int64
		specReplicas int32
		status       extensions.DeploymentStatus
		msg          string
		done         bool
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
			generation:   1,
			specReplicas: 2,
			status: extensions.DeploymentStatus{
				ObservedGeneration:  1,
				Replicas:            2,
				UpdatedReplicas:     2,
				AvailableReplicas:   1,
				UnavailableReplicas: 1,
			},

			msg:  "Waiting for rollout to finish: 1 of 2 updated replicas are available...\n",
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
	}

	for _, test := range tests {
		d := &extensions.Deployment{
			ObjectMeta: api.ObjectMeta{
				Namespace:  "bar",
				Name:       "foo",
				UID:        "8764ae47-9092-11e4-8393-42010af018ff",
				Generation: test.generation,
			},
			Spec: extensions.DeploymentSpec{
				Replicas: test.specReplicas,
			},
			Status: test.status,
		}
		client := fake.NewSimpleClientset(d).Extensions()
		dsv := &DeploymentStatusViewer{c: client}
		msg, done, err := dsv.Status("bar", "foo", 0)
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
	}
}
