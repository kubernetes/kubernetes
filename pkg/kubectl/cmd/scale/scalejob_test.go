/*
Copyright 2018 The Kubernetes Authors.

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

package scale

import (
	"errors"
	"testing"

	batch "k8s.io/api/batch/v1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/client-go/kubernetes/fake"
	batchclient "k8s.io/client-go/kubernetes/typed/batch/v1"
	testcore "k8s.io/client-go/testing"
)

type errorJobs struct {
	batchclient.JobInterface
	conflict bool
	invalid  bool
}

func (c *errorJobs) Update(job *batch.Job) (*batch.Job, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(job.Kind), job.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(job.Kind), job.Name, nil)
	}
	return nil, errors.New("Job update failure")
}

func (c *errorJobs) Get(name string, options metav1.GetOptions) (*batch.Job, error) {
	zero := int32(0)
	return &batch.Job{
		Spec: batch.JobSpec{
			Parallelism: &zero,
		},
	}, nil
}

type errorJobClient struct {
	batchclient.JobsGetter
	conflict bool
	invalid  bool
}

func (c *errorJobClient) Jobs(namespace string) batchclient.JobInterface {
	return &errorJobs{
		JobInterface: c.JobsGetter.Jobs(namespace),
		conflict:     c.conflict,
		invalid:      c.invalid,
	}
}

func TestJobScaleRetry(t *testing.T) {
	fake := &errorJobClient{JobsGetter: fake.NewSimpleClientset().Batch(), conflict: true}
	scaler := &JobPsuedoScaler{JobsClient: fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := scaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = scaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err = scaleFunc()
	if err == nil {
		t.Error("Expected error on precondition failure")
	}
}

func job() *batch.Job {
	return &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
	}
}

func TestJobScale(t *testing.T) {
	fakeClientset := fake.NewSimpleClientset(job())
	scaler := &JobPsuedoScaler{JobsClient: fakeClientset.Batch()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fakeClientset.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != batch.Resource("jobs") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-job %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != batch.Resource("jobs") || *action.GetObject().(*batch.Job).Spec.Parallelism != int32(count) {
		t.Errorf("unexpected action %v, expected update-job with parallelism = %d", actions[1], count)
	}
}

func TestJobScaleInvalid(t *testing.T) {
	fake := &errorJobClient{JobsGetter: fake.NewSimpleClientset().Batch(), invalid: true}
	scaler := &JobPsuedoScaler{JobsClient: fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := scaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err == nil {
		t.Errorf("Expected error on invalid update failure, got %v", err)
	}
}

func TestJobScaleFailsPreconditions(t *testing.T) {
	ten := int32(10)
	fake := fake.NewSimpleClientset(&batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
		Spec: batch.JobSpec{
			Parallelism: &ten,
		},
	})
	scaler := &JobPsuedoScaler{JobsClient: fake.Batch()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 actions (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != batch.Resource("jobs") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-job %s", actions[0], name)
	}
}

func TestValidateJob(t *testing.T) {
	zero, ten, twenty := int32(0), int32(10), int32(20)
	tests := []struct {
		preconditions ScalePrecondition
		job           batch.Job
		expectError   bool
		test          string
	}{
		{
			preconditions: ScalePrecondition{-1, ""},
			expectError:   false,
			test:          "defaults",
		},
		{
			preconditions: ScalePrecondition{-1, ""},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: batch.JobSpec{
					Parallelism: &ten,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: batch.JobSpec{
					Parallelism: &zero,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: batch.JobSpec{
					Parallelism: &ten,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: batch.JobSpec{
					Parallelism: &ten,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: batch.JobSpec{
					Parallelism: &twenty,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
			},
			expectError: true,
			test:        "parallelism nil",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: batch.JobSpec{
					Parallelism: &ten,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: batch.JobSpec{
					Parallelism: &twenty,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := validateJob(&test.job, &test.preconditions)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}
