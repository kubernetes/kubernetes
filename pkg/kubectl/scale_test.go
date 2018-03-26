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
	"fmt"
	"testing"
	"time"

	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/scale"
	testcore "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
)

func TestReplicationControllerScaleRetry(t *testing.T) {
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"get", "update", "get"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "", Resource: "replicationcontrollers"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	namespace := metav1.NamespaceDefault

	scaleFunc := ScaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update conflict failure, got %v", err)
	}
	preconditions = ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, &preconditions, namespace, name, count, nil)
	pass, err = scaleFunc()
	if err == nil {
		t.Errorf("Expected error on precondition failure")
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicationControllerScaleInvalid(t *testing.T) {
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 1, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "", Resource: "replicationcontrollers"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicationControllerScale(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 2, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "", Resource: "replicationcontrollers"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo-v1"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)

	if err != nil {
		t.Fatalf("unexpected error occurred = %v while scaling the resource", err)
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicationControllerScaleFailsPreconditions(t *testing.T) {
	scaleClientExpectedAction := []string{"get"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo", 10, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "", Resource: "replicationcontrollers"})
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err == nil {
		t.Fatal("expected to get an error but none was returned")
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
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

func TestDeploymentScaleRetry(t *testing.T) {
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"get", "update", "get"}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "deployments"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestDeploymentScale(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "deployments"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestDeploymentScaleInvalid(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "deployments"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestDeploymentScaleFailsPreconditions(t *testing.T) {
	scaleClientExpectedAction := []string{"get"}
	scaleClient := createFakeScaleClient("deployments", "foo", 10, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "deployments"})
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err == nil {
		t.Fatal("exptected to get an error but none was returned")
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestStatefulSetScale(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "statefullset"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestStatefulSetScaleRetry(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update", "get"}
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "statefulsets"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestStatefulSetScaleInvalid(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "statefulsets"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestStatefulSetScaleFailsPreconditions(t *testing.T) {
	scaleClientExpectedAction := []string{"get"}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 10, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "apps", Resource: "statefulsets"})
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err == nil {
		t.Fatal("expected to get an error but none was returned")
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicaSetScale(t *testing.T) {
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 10, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "extensions", Resource: "replicasets"})
	preconditions := ScalePrecondition{-1, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicaSetScaleRetry(t *testing.T) {
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"get", "update", "get"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "extensions", Resource: "replicasets"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicaSetScaleInvalid(t *testing.T) {
	verbsOnError := map[string]*kerrors.StatusError{
		"update": kerrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"get", "update"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "extensions", Resource: "replicasets"})
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
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

func TestReplicaSetsGetterFailsPreconditions(t *testing.T) {
	scaleClientExpectedAction := []string{"get"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 10, nil)
	scaler := NewScaler(scaleClient, schema.GroupResource{Group: "extensions", Resource: "replicasets"})
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil)
	if err == nil {
		t.Fatal("expected to get an error but non was returned")
	}
	actions := scaleClient.Actions()
	if len(actions) != len(scaleClientExpectedAction) {
		t.Errorf("unexpected actions: %v, expected %d actions got %d", actions, len(scaleClientExpectedAction), len(actions))
	}
	for i, verb := range scaleClientExpectedAction {
		if actions[i].GetVerb() != verb {
			t.Errorf("unexpected action: %+v, expected %s", actions[i].GetVerb(), verb)
		}
	}
}

// TestGenericScaleSimple exercises GenericScaler.ScaleSimple method
func TestGenericScaleSimple(t *testing.T) {
	// test data
	scaleClient := createFakeScaleClient("deployments", "abc", 10, nil)

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
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 2: scale down the "abc" deployment
		{
			name:         "scale down the \"abs\" deployment",
			precondition: ScalePrecondition{20, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 3: precondition error, expected size is 1,
		// note that the previous scenario (2) set the size to 5
		{
			name:         "precondition error, expected size is 1",
			precondition: ScalePrecondition{1, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 4: precondition is not validated when the precondition size is set to -1
		{
			name:         "precondition is not validated when the size is set to -1",
			precondition: ScalePrecondition{-1, ""},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 5: precondition error, resource version mismatch
		{
			name:         "precondition error, resource version mismatch",
			precondition: ScalePrecondition{5, "v1"},
			newSize:      5,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("running scenario %d: %s", index+1, scenario.name), func(t *testing.T) {
			target := NewScaler(scenario.scaleGetter, scenario.targetGR)

			resVersion, err := target.ScaleSimple("default", scenario.resName, &scenario.precondition, uint(scenario.newSize))

			if scenario.expectError && err == nil {
				t.Fatal("expected an error but was not returned")
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
	scaleClient := createFakeScaleClient("deployments", "abc", 10, nil)

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
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 2: a resource name cannot be empty
		{
			name:         "a resource name cannot be empty",
			precondition: ScalePrecondition{10, ""},
			newSize:      20,
			targetGR:     schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:      "",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 3: wait for replicas error due to status.Replicas != spec.Replicas
		{
			name:            "wait for replicas error due to status.Replicas != spec.Replicas",
			precondition:    ScalePrecondition{10, ""},
			newSize:         20,
			targetGR:        schema.GroupResource{Group: "apps", Resource: "deployments"},
			resName:         "abc",
			scaleGetter:     scaleClient,
			waitForReplicas: &RetryParams{time.Duration(5 * time.Second), time.Duration(5 * time.Second)},
			expectError:     true,
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("running scenario %d: %s", index+1, scenario.name), func(t *testing.T) {
			target := NewScaler(scenario.scaleGetter, scenario.targetGR)

			err := target.Scale("default", scenario.resName, uint(scenario.newSize), &scenario.precondition, nil, scenario.waitForReplicas)

			if scenario.expectError && err == nil {
				t.Fatal("expected an error but was not returned")
			}
			if !scenario.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
