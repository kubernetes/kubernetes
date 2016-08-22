/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kerrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	testcore "k8s.io/kubernetes/pkg/client/testing/core"
)

type ErrorScales struct {
	extensionsclient.ScaleInterface
	conflict bool
	invalid  bool
}

func (c *ErrorScales) Update(kind string, scale *extensions.Scale) (*extensions.Scale, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(kind), scale.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(kind), scale.Name, nil)
	}
	return nil, errors.New("Scale update failure")
}

type ErrorExtensions struct {
	*fake.Clientset
	conflict bool
	invalid  bool
}

func (c *ErrorExtensions) Scales(namespace string) extensionsclient.ScaleInterface {
	return &ErrorScales{
		ScaleInterface: c.Clientset.Extensions().Scales(namespace),
		conflict:       c.conflict,
		invalid:        c.invalid,
	}
}

func scale() *extensions.Scale {
	return &extensions.Scale{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
			Name:      "foo-v1",
		},
		Spec: extensions.ScaleSpec{
			Replicas: 0,
		},
	}
}

func TestReplicationControllerScaleRetry(t *testing.T) {
	clientset := fake.NewSimpleClientset(oldRc(0, 0), scale())
	scaler := ReplicationControllerScaler{clientset.Core(), &ErrorExtensions{Clientset: clientset, conflict: true}}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := api.NamespaceDefault

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update conflict failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err = scaleFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
}

func TestReplicationControllerScaleInvalid(t *testing.T) {
	clientset := fake.NewSimpleClientset(oldRc(0, 0), scale())
	scaler := ReplicationControllerScaler{clientset.Core(), &ErrorExtensions{Clientset: clientset, invalid: true}}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := "default"

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	e, ok := err.(ScaleError)
	if err == nil || !ok || e.FailureType != ScaleUpdateFailure {
		t.Errorf("Expected error on invalid update failure, got %v", err)
	}
}

func TestReplicationControllerScale(t *testing.T) {
	fake := fake.NewSimpleClientset(oldRc(0, 0), scale())
	scaler := ReplicationControllerScaler{fake.Core(), fake.Extensions()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("ReplicationController") || action.GetName() != name || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action: %v, expected get-ReplicationController %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != api.Resource("ReplicationController") || action.GetObject().(*extensions.Scale).Spec.Replicas != int32(count) || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action %v, expected update-ReplicationController with replicas = %d", actions[1], count)
	}
}

func TestReplicationControllerScaleFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(
		&api.ReplicationController{
			ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
			Spec: api.ReplicationControllerSpec{
				Replicas: 10,
			},
		},
		&extensions.Scale{
			ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo"},
			Spec: extensions.ScaleSpec{
				Replicas: 10,
			},
		},
	)
	scaler := ReplicationControllerScaler{fake.Core(), fake.Extensions()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 action (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("ReplicationController") || action.GetName() != name || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action: %v, expected get-ReplicationController %s", actions[0], name)
	}
}

func TestValidateScale(t *testing.T) {
	tests := []struct {
		preconditions ScalePrecondition
		scale         extensions.Scale
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
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 0,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 10,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			scale: extensions.Scale{
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.ScaleSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.ValidateScale(&test.scale)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error: %v (%s)", err, test.test)
		}
	}
}

type ErrorJobs struct {
	batchclient.JobInterface
	conflict bool
	invalid  bool
}

func (c *ErrorJobs) Update(job *batch.Job) (*batch.Job, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(job.Kind), job.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(job.Kind), job.Name, nil)
	}
	return nil, errors.New("Job update failure")
}

func (c *ErrorJobs) Get(name string) (*batch.Job, error) {
	zero := int32(0)
	return &batch.Job{
		Spec: batch.JobSpec{
			Parallelism: &zero,
		},
	}, nil
}

type ErrorJobClient struct {
	batchclient.JobsGetter
	conflict bool
	invalid  bool
}

func (c *ErrorJobClient) Jobs(namespace string) batchclient.JobInterface {
	return &ErrorJobs{
		JobInterface: c.JobsGetter.Jobs(namespace),
		conflict:     c.conflict,
		invalid:      c.invalid,
	}
}

func TestJobScaleRetry(t *testing.T) {
	fake := &ErrorJobClient{JobsGetter: fake.NewSimpleClientset().Batch(), conflict: true}
	scaler := JobScaler{fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err = scaleFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
}

func job() *batch.Job {
	return &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
			Name:      "foo",
		},
	}
}

func TestJobScale(t *testing.T) {
	fakeClientset := fake.NewSimpleClientset(job())
	scaler := JobScaler{fakeClientset.Batch()}
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
	fake := &ErrorJobClient{JobsGetter: fake.NewSimpleClientset().Batch(), invalid: true}
	scaler := JobScaler{fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	e, ok := err.(ScaleError)
	if err == nil || !ok || e.FailureType != ScaleUpdateFailure {
		t.Errorf("Expected error on invalid update failure, got %v", err)
	}
}

func TestJobScaleFailsPreconditions(t *testing.T) {
	ten := int32(10)
	fake := fake.NewSimpleClientset(&batch.Job{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
			Name:      "foo",
		},
		Spec: batch.JobSpec{
			Parallelism: &ten,
		},
	})
	scaler := JobScaler{fake.Batch()}
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
					ResourceVersion: "foo",
				},
			},
			expectError: true,
			test:        "parallelism nil",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			job: batch.Job{
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{
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
		err := test.preconditions.ValidateJob(&test.job)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error: %v (%s)", err, test.test)
		}
	}
}

func deployment() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
			Name:      "foo-v1",
		},
	}
}

func TestDeploymentScaleRetry(t *testing.T) {
	clientset := fake.NewSimpleClientset(deployment(), scale())
	fake := &ErrorExtensions{Clientset: clientset, conflict: true}
	scaler := &DeploymentScaler{clientset.Extensions(), fake}
	preconditions := &ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = &ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, preconditions, namespace, name, count, nil)
	pass, err = scaleFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
}

func TestDeploymentScale(t *testing.T) {
	fake := fake.NewSimpleClientset(deployment(), scale())
	scaler := DeploymentScaler{fake.Extensions(), fake.Extensions()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("Deployment") || action.GetName() != name || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action: %v, expected get-Deployment %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != api.Resource("Deployment") || action.GetObject().(*extensions.Scale).Spec.Replicas != int32(count) || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action %v, expected update-Deployment with replicas = %d", actions[1], count)
	}
}

func TestDeploymentScaleInvalid(t *testing.T) {
	clientset := fake.NewSimpleClientset(deployment(), scale())
	fake := &ErrorExtensions{Clientset: clientset, invalid: true}
	scaler := DeploymentScaler{clientset.Extensions(), fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := "default"

	scaleFunc := ScaleCondition(&scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	e, ok := err.(ScaleError)
	if err == nil || !ok || e.FailureType != ScaleUpdateFailure {
		t.Errorf("Expected error on invalid update failure, got %v", err)
	}
}

func TestDeploymentScaleFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(&extensions.Deployment{
		ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault, Name: "foo-v1"},
		Spec: extensions.DeploymentSpec{
			Replicas: 10,
		},
	}, scale())
	scaler := DeploymentScaler{fake.Extensions(), fake.Extensions()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo-v1"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 actions (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("Deployment") || action.GetName() != name || action.GetSubresource() != "scale" {
		t.Errorf("unexpected action: %v, expected get-Deployment %s", actions[0], name)
	}
}
