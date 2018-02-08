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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/discovery"
	fakedisco "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/dynamic"
	fakerest "k8s.io/client-go/rest/fake"
	"k8s.io/client-go/scale"
	testcore "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	appsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
)

type ErrorReplicationControllers struct {
	coreclient.ReplicationControllerInterface
	conflict bool
	invalid  bool
}

func (c *ErrorReplicationControllers) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(controller.Kind), controller.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(controller.Kind), controller.Name, nil)
	}
	return nil, errors.New("Replication controller update failure")
}

type ErrorReplicationControllerClient struct {
	*fake.Clientset
	conflict bool
	invalid  bool
}

func (c *ErrorReplicationControllerClient) ReplicationControllers(namespace string) coreclient.ReplicationControllerInterface {
	return &ErrorReplicationControllers{
		ReplicationControllerInterface: c.Clientset.Core().ReplicationControllers(namespace),
		conflict:                       c.conflict,
		invalid:                        c.invalid,
	}
}

func TestReplicationControllerScaleRetry(t *testing.T) {
	fake := &ErrorReplicationControllerClient{Clientset: fake.NewSimpleClientset(oldRc(0, 0)), conflict: true}
	scaler := ReplicationControllerScaler{fake}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := metav1.NamespaceDefault

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
	fake := &ErrorReplicationControllerClient{Clientset: fake.NewSimpleClientset(oldRc(0, 0)), invalid: true}
	scaler := ReplicationControllerScaler{fake}
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
	fake := fake.NewSimpleClientset(oldRc(0, 0))
	scaler := ReplicationControllerScaler{fake.Core()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("replicationcontrollers") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != api.Resource("replicationcontrollers") || action.GetObject().(*api.ReplicationController).Spec.Replicas != int32(count) {
		t.Errorf("unexpected action %v, expected update-replicationController with replicas = %d", actions[1], count)
	}
}

func TestReplicationControllerScaleFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(&api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault, Name: "foo"},
		Spec: api.ReplicationControllerSpec{
			Replicas: 10,
		},
	})
	scaler := ReplicationControllerScaler{fake.Core()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 action (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != api.Resource("replicationcontrollers") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicationController %s", actions[0], name)
	}
}

func TestValidateReplicationController(t *testing.T) {
	tests := []struct {
		preconditions ScalePrecondition
		controller    api.ReplicationController
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
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 0,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 10,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			controller: api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 20,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.ValidateReplicationController(&test.controller)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}

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
	scaler := ScalerFor(schema.GroupKind{Group: batch.GroupName, Kind: "Job"}, fake, nil, schema.GroupResource{})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, &preconditions, namespace, name, count, nil)
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
	scaler := ScalerFor(schema.GroupKind{Group: batch.GroupName, Kind: "Job"}, fakeClientset.Batch(), nil, schema.GroupResource{})
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
	scaler := ScalerFor(schema.GroupKind{Group: batch.GroupName, Kind: "Job"}, fake, nil, schema.GroupResource{})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, &preconditions, namespace, name, count, nil)
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
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
		Spec: batch.JobSpec{
			Parallelism: &ten,
		},
	})
	scaler := ScalerFor(schema.GroupKind{Group: batch.GroupName, Kind: "Job"}, fake.Batch(), nil, schema.GroupResource{})
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
		err := test.preconditions.ValidateJob(&test.job)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}

type ErrorDeployments struct {
	extensionsclient.DeploymentInterface
	conflict bool
	invalid  bool
}

func (c *ErrorDeployments) Update(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(deployment.Kind), deployment.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(deployment.Kind), deployment.Name, nil)
	}
	return nil, errors.New("deployment update failure")
}

func (c *ErrorDeployments) Get(name string, options metav1.GetOptions) (*extensions.Deployment, error) {
	return &extensions.Deployment{
		Spec: extensions.DeploymentSpec{
			Replicas: 0,
		},
	}, nil
}

type ErrorDeploymentClient struct {
	extensionsclient.DeploymentsGetter
	conflict bool
	invalid  bool
}

func (c *ErrorDeploymentClient) Deployments(namespace string) extensionsclient.DeploymentInterface {
	return &ErrorDeployments{
		DeploymentInterface: c.DeploymentsGetter.Deployments(namespace),
		invalid:             c.invalid,
		conflict:            c.conflict,
	}
}

func TestDeploymentScaleRetry(t *testing.T) {
	fake := &ErrorDeploymentClient{DeploymentsGetter: fake.NewSimpleClientset().Extensions(), conflict: true}
	scaler := &DeploymentScaler{fake}
	preconditions := &ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
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
		t.Error("Expected error on precondition failure")
	}
}

func deployment() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
	}
}

func TestDeploymentScale(t *testing.T) {
	fake := fake.NewSimpleClientset(deployment())
	scaler := DeploymentScaler{fake.Extensions()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != extensions.Resource("deployments") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-deployment %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != extensions.Resource("deployments") || action.GetObject().(*extensions.Deployment).Spec.Replicas != int32(count) {
		t.Errorf("unexpected action %v, expected update-deployment with replicas = %d", actions[1], count)
	}
}

func TestDeploymentScaleInvalid(t *testing.T) {
	fake := &ErrorDeploymentClient{DeploymentsGetter: fake.NewSimpleClientset().Extensions(), invalid: true}
	scaler := DeploymentScaler{fake}
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

func TestDeploymentScaleFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(&extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 10,
		},
	})
	scaler := DeploymentScaler{fake.Extensions()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 actions (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != extensions.Resource("deployments") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-deployment %s", actions[0], name)
	}
}

func TestValidateDeployment(t *testing.T) {
	zero, ten, twenty := int32(0), int32(10), int32(20)
	tests := []struct {
		preconditions ScalePrecondition
		deployment    extensions.Deployment
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
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: zero,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
			},
			expectError: true,
			test:        "no replicas",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: ten,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			deployment: extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.DeploymentSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.ValidateDeployment(&test.deployment)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}

type ErrorStatefulSets struct {
	appsclient.StatefulSetInterface
	conflict bool
	invalid  bool
}

func (c *ErrorStatefulSets) Update(statefulSet *apps.StatefulSet) (*apps.StatefulSet, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(statefulSet.Kind), statefulSet.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(statefulSet.Kind), statefulSet.Name, nil)
	}
	return nil, errors.New("statefulSet update failure")
}

func (c *ErrorStatefulSets) Get(name string, options metav1.GetOptions) (*apps.StatefulSet, error) {
	return &apps.StatefulSet{
		Spec: apps.StatefulSetSpec{
			Replicas: 0,
		},
	}, nil
}

type ErrorStatefulSetClient struct {
	appsclient.StatefulSetsGetter
	conflict bool
	invalid  bool
}

func (c *ErrorStatefulSetClient) StatefulSets(namespace string) appsclient.StatefulSetInterface {
	return &ErrorStatefulSets{
		StatefulSetInterface: c.StatefulSetsGetter.StatefulSets(namespace),
		invalid:              c.invalid,
		conflict:             c.conflict,
	}
}

func statefulSet() *apps.StatefulSet {
	return &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
	}
}

func TestStatefulSetScale(t *testing.T) {
	fake := fake.NewSimpleClientset(statefulSet())
	scaler := StatefulSetScaler{fake.Apps()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}

	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != apps.Resource("statefulsets") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-statefulsets %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != apps.Resource("statefulsets") || action.GetObject().(*apps.StatefulSet).Spec.Replicas != int32(count) {
		t.Errorf("unexpected action %v, expected update-statefulset with replicas = %d", actions[1], count)
	}
}

func TestStatefulSetScaleRetry(t *testing.T) {
	fake := &ErrorStatefulSetClient{StatefulSetsGetter: fake.NewSimpleClientset().Apps(), conflict: true}
	scaler := &StatefulSetScaler{fake}
	preconditions := &ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
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
		t.Error("Expected error on precondition failure")
	}
}

func TestStatefulSetScaleInvalid(t *testing.T) {
	fake := &ErrorStatefulSetClient{StatefulSetsGetter: fake.NewSimpleClientset().Apps(), invalid: true}
	scaler := StatefulSetScaler{fake}
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

func TestStatefulSetScaleFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(&apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
		Spec: apps.StatefulSetSpec{
			Replicas: 10,
		},
	})
	scaler := StatefulSetScaler{fake.Apps()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 actions (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != apps.Resource("statefulsets") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-statefulset %s", actions[0], name)
	}
}

func TestValidateStatefulSet(t *testing.T) {
	zero, ten, twenty := int32(0), int32(10), int32(20)
	tests := []struct {
		preconditions ScalePrecondition
		statefulset   apps.StatefulSet
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
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: zero,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
			},
			expectError: true,
			test:        "no replicas",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: ten,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			statefulset: apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.ValidateStatefulSet(&test.statefulset)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}

type ErrorReplicaSets struct {
	extensionsclient.ReplicaSetInterface
	conflict bool
	invalid  bool
}

func (c *ErrorReplicaSets) Update(replicaSets *extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	switch {
	case c.invalid:
		return nil, kerrors.NewInvalid(api.Kind(replicaSets.Kind), replicaSets.Name, nil)
	case c.conflict:
		return nil, kerrors.NewConflict(api.Resource(replicaSets.Kind), replicaSets.Name, nil)
	}
	return nil, errors.New("replicaSets update failure")
}

func (c *ErrorReplicaSets) Get(name string, options metav1.GetOptions) (*extensions.ReplicaSet, error) {
	return &extensions.ReplicaSet{
		Spec: extensions.ReplicaSetSpec{
			Replicas: 0,
		},
	}, nil
}

type ErrorReplicaSetClient struct {
	extensionsclient.ReplicaSetsGetter
	conflict bool
	invalid  bool
}

func (c *ErrorReplicaSetClient) ReplicaSets(namespace string) extensionsclient.ReplicaSetInterface {
	return &ErrorReplicaSets{
		ReplicaSetInterface: c.ReplicaSetsGetter.ReplicaSets(namespace),
		invalid:             c.invalid,
		conflict:            c.conflict,
	}
}

func replicaSet() *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
	}
}

func TestReplicaSetScale(t *testing.T) {
	fake := fake.NewSimpleClientset(replicaSet())
	scaler := ReplicaSetScaler{fake.Extensions()}
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v, expected 2 actions (get, update)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != extensions.Resource("replicasets") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicationSet %s", actions[0], name)
	}
	if action, ok := actions[1].(testcore.UpdateAction); !ok || action.GetResource().GroupResource() != extensions.Resource("replicasets") || action.GetObject().(*extensions.ReplicaSet).Spec.Replicas != int32(count) {
		t.Errorf("unexpected action %v, expected update-replicaSet with replicas = %d", actions[1], count)
	}
}

func TestReplicaSetScaleRetry(t *testing.T) {
	fake := &ErrorReplicaSetClient{ReplicaSetsGetter: fake.NewSimpleClientset().Extensions(), conflict: true}
	scaler := &ReplicaSetScaler{fake}
	preconditions := &ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
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
		t.Error("Expected error on precondition failure")
	}
}

func TestReplicaSetScaleInvalid(t *testing.T) {
	fake := &ErrorReplicaSetClient{ReplicaSetsGetter: fake.NewSimpleClientset().Extensions(), invalid: true}
	scaler := ReplicaSetScaler{fake}
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

func TestReplicaSetsGetterFailsPreconditions(t *testing.T) {
	fake := fake.NewSimpleClientset(&extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      "foo",
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: 10,
		},
	})
	scaler := ReplicaSetScaler{fake.Extensions()}
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	scaler.Scale("default", name, count, &preconditions, nil, nil)

	actions := fake.Actions()
	if len(actions) != 1 {
		t.Errorf("unexpected actions: %v, expected 1 actions (get)", actions)
	}
	if action, ok := actions[0].(testcore.GetAction); !ok || action.GetResource().GroupResource() != extensions.Resource("replicasets") || action.GetName() != name {
		t.Errorf("unexpected action: %v, expected get-replicaSets %s", actions[0], name)
	}
}

func TestValidateReplicaSets(t *testing.T) {
	zero, ten, twenty := int32(0), int32(10), int32(20)
	tests := []struct {
		preconditions ScalePrecondition
		replicaSets   extensions.ReplicaSet
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
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "defaults 2",
		},
		{
			preconditions: ScalePrecondition{0, ""},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: zero,
				},
			},
			expectError: false,
			test:        "size matches",
		},
		{
			preconditions: ScalePrecondition{-1, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "resource version matches",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: ten,
				},
			},
			expectError: false,
			test:        "both match",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "size different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "foo",
				},
			},
			expectError: true,
			test:        "no replicas",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: ten,
				},
			},
			expectError: true,
			test:        "version different",
		},
		{
			preconditions: ScalePrecondition{10, "foo"},
			replicaSets: extensions.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "bar",
				},
				Spec: extensions.ReplicaSetSpec{
					Replicas: twenty,
				},
			},
			expectError: true,
			test:        "both different",
		},
	}
	for _, test := range tests {
		err := test.preconditions.ValidateReplicaSet(&test.replicaSets)
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if err == nil && test.expectError {
			t.Errorf("expected an error: %v (%s)", err, test.test)
		}
	}
}

// TestGenericScaleSimple exercises GenericScaler.ScaleSimple method
func TestGenericScaleSimple(t *testing.T) {
	// test data
	discoveryResources := []*metav1.APIResourceList{
		{
			GroupVersion: appsv1beta2.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{Name: "deployments", Namespaced: true, Kind: "Deployment"},
				{Name: "deployments/scale", Namespaced: true, Kind: "Scale", Group: "apps", Version: "v1beta2"},
			},
		},
	}
	appsV1beta2Scale := &appsv1beta2.Scale{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Scale",
			APIVersion: appsv1beta2.SchemeGroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "abc",
		},
		Spec: appsv1beta2.ScaleSpec{Replicas: 10},
		Status: appsv1beta2.ScaleStatus{
			Replicas: 10,
		},
	}
	pathsResources := map[string]runtime.Object{
		"/apis/apps/v1beta2/namespaces/default/deployments/abc/scale": appsV1beta2Scale,
	}

	scaleClient, err := fakeScaleClient(discoveryResources, pathsResources)
	if err != nil {
		t.Fatal(err)
	}

	// test scenarios
	scenarios := []struct {
		name         string
		precondition ScalePrecondition
		newSize      int
		targetGR     schema.GroupResource
		resName      string
		scaleGetter  scale.ScalesGetter
		expectError  bool
	}{
		// scenario 1: scale up the "abc" deployment
		{
			name:         "scale up the \"abc\" deployment",
			precondition: ScalePrecondition{10, ""},
			newSize:      20,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 2: scale down the "abc" deployment
		{
			name:         "scale down the \"abs\" deplyment",
			precondition: ScalePrecondition{20, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 3: precondition error, expected size is 1,
		// note that the previous scenario (2) set the size to 5
		{
			name:         "precondition error, expected size is 1",
			precondition: ScalePrecondition{1, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 4: precondition is not validated when the precondition size is set to -1
		{
			name:         "precondition is not validated when the size is set to -1",
			precondition: ScalePrecondition{-1, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 5: precondition error, resource version mismatch
		{
			name:         "precondition error, resource version mismatch",
			precondition: ScalePrecondition{5, "v1"},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("running scenario %d: %s", index+1, scenario.name), func(t *testing.T) {
			target := ScalerFor(schema.GroupKind{}, nil, scenario.scaleGetter, scenario.targetGR)

			resVersion, err := target.ScaleSimple("default", scenario.resName, &scenario.precondition, uint(scenario.newSize))

			if scenario.expectError && err == nil {
				t.Fatal("expeced an error but was not returned")
			}
			if !scenario.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resVersion != "" {
				t.Fatalf("unexpected resource version returned = %s, wanted = %s", resVersion, "")
			}
		})
	}
}

// TestGenericScale exercises GenericScaler.Scale method
func TestGenericScale(t *testing.T) {
	// test data
	discoveryResources := []*metav1.APIResourceList{
		{
			GroupVersion: appsv1beta2.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{Name: "deployments", Namespaced: true, Kind: "Deployment"},
				{Name: "deployments/scale", Namespaced: true, Kind: "Scale", Group: "apps", Version: "v1beta2"},
			},
		},
	}
	appsV1beta2Scale := &appsv1beta2.Scale{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Scale",
			APIVersion: appsv1beta2.SchemeGroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "abc",
		},
		Spec: appsv1beta2.ScaleSpec{Replicas: 10},
		Status: appsv1beta2.ScaleStatus{
			Replicas: 10,
		},
	}
	pathsResources := map[string]runtime.Object{
		"/apis/apps/v1beta2/namespaces/default/deployments/abc/scale": appsV1beta2Scale,
	}

	scaleClient, err := fakeScaleClient(discoveryResources, pathsResources)
	if err != nil {
		t.Fatal(err)
	}

	// test scenarios
	scenarios := []struct {
		name            string
		precondition    ScalePrecondition
		newSize         int
		targetGR        schema.GroupResource
		resName         string
		scaleGetter     scale.ScalesGetter
		waitForReplicas *RetryParams
		expectError     bool
	}{
		// scenario 1: scale up the "abc" deployment
		{
			name:         "scale up the \"abc\" deployment",
			precondition: ScalePrecondition{10, ""},
			newSize:      20,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 2: a resource name cannot be empty
		{
			name:         "a resource name cannot be empty",
			precondition: ScalePrecondition{10, ""},
			newSize:      20,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:      "",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 3: wait for replicas error due to status.Replicas != spec.Replicas
		{
			name:            "wait for replicas error due to status.Replicas != spec.Replicas",
			precondition:    ScalePrecondition{10, ""},
			newSize:         20,
			targetGR:        schema.GroupResource{Group: "apps", Resource: "deployment"},
			resName:         "abc",
			scaleGetter:     scaleClient,
			waitForReplicas: &RetryParams{time.Duration(5 * time.Second), time.Duration(5 * time.Second)},
			expectError:     true,
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("running scenario %d: %s", index+1, scenario.name), func(t *testing.T) {
			target := ScalerFor(schema.GroupKind{}, nil, scenario.scaleGetter, scenario.targetGR)

			err := target.Scale("default", scenario.resName, uint(scenario.newSize), &scenario.precondition, nil, scenario.waitForReplicas)

			if scenario.expectError && err == nil {
				t.Fatal("expeced an error but was not returned")
			}
			if !scenario.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func fakeScaleClient(discoveryResources []*metav1.APIResourceList, pathsResources map[string]runtime.Object) (scale.ScalesGetter, error) {
	fakeDiscoveryClient := &fakedisco.FakeDiscovery{Fake: &testcore.Fake{}}
	fakeDiscoveryClient.Resources = discoveryResources
	restMapperRes, err := discovery.GetAPIGroupResources(fakeDiscoveryClient)
	if err != nil {
		return nil, err
	}
	restMapper := discovery.NewRESTMapper(restMapperRes, apimeta.InterfacesForUnstructured)
	codecs := serializer.NewCodecFactory(scale.NewScaleConverter().Scheme())
	fakeReqHandler := func(req *http.Request) (*http.Response, error) {
		path := req.URL.Path
		scale, isScalePath := pathsResources[path]
		if !isScalePath {
			return nil, fmt.Errorf("unexpected request for URL %q with method %q", req.URL.String(), req.Method)
		}

		switch req.Method {
		case "GET":
			res, err := json.Marshal(scale)
			if err != nil {
				return nil, err
			}
			return &http.Response{StatusCode: 200, Header: defaultHeaders(), Body: bytesBody(res)}, nil
		case "PUT":
			decoder := codecs.UniversalDeserializer()
			body, err := ioutil.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
			newScale, newScaleGVK, err := decoder.Decode(body, nil, nil)
			if err != nil {
				return nil, fmt.Errorf("unexpected request body: %v", err)
			}
			if *newScaleGVK != scale.GetObjectKind().GroupVersionKind() {
				return nil, fmt.Errorf("unexpected scale API version %s (expected %s)", newScaleGVK.String(), scale.GetObjectKind().GroupVersionKind().String())
			}
			res, err := json.Marshal(newScale)
			if err != nil {
				return nil, err
			}

			pathsResources[path] = newScale
			return &http.Response{StatusCode: 200, Header: defaultHeaders(), Body: bytesBody(res)}, nil
		default:
			return nil, fmt.Errorf("unexpected request for URL %q with method %q", req.URL.String(), req.Method)
		}
	}

	fakeClient := &fakerest.RESTClient{
		Client: fakerest.CreateHTTPClient(fakeReqHandler),
		NegotiatedSerializer: serializer.DirectCodecFactory{
			CodecFactory: serializer.NewCodecFactory(scale.NewScaleConverter().Scheme()),
		},
		GroupVersion:     schema.GroupVersion{},
		VersionedAPIPath: "/not/a/real/path",
	}

	resolver := scale.NewDiscoveryScaleKindResolver(fakeDiscoveryClient)
	client := scale.New(fakeClient, restMapper, dynamic.LegacyAPIPathResolverFunc, resolver)
	return client, nil
}

func bytesBody(bodyBytes []byte) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader(bodyBytes))
}

func defaultHeaders() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}
