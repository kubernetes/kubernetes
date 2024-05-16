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

package scale

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/scale"
	fakescale "k8s.io/client-go/scale/fake"
	testcore "k8s.io/client-go/testing"
)

var (
	rcgvr = schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "replicationcontrollers",
	}
	rsgvr = schema.GroupVersionResource{
		Group:    "extensions",
		Version:  "v1beta1",
		Resource: "replicasets",
	}
	deploygvr = schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	stsgvr = schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "statefulsets",
	}
)

func TestReplicationControllerScaleRetry(t *testing.T) {
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"patch", "get"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo-v1"
	namespace := metav1.NamespaceDefault

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, rcgvr, false)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update conflict failure, got %v", err)
	}
	preconditions := ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, &preconditions, namespace, name, count, nil, rcgvr, false)
	_, err = scaleFunc()
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
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 1, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo-v1"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, rcgvr, false)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err == nil {
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
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("replicationcontrollers", "foo-v1", 2, nil)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo-v1"
	err := scaler.Scale("default", name, count, nil, nil, nil, rcgvr, false)

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
	scaler := NewScaler(scaleClient)
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil, rcgvr, false)
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

func TestDeploymentScaleRetry(t *testing.T) {
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"patch", "get"}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, deploygvr, false)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions := &ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, preconditions, namespace, name, count, nil, deploygvr, false)
	_, err = scaleFunc()
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
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, nil)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, nil, nil, nil, deploygvr, false)
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
	scaleClientExpectedAction := []string{"patch"}
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("deployments", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, deploygvr, false)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err == nil {
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
	scaler := NewScaler(scaleClient)
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil, deploygvr, false)
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
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, nil)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, nil, nil, nil, stsgvr, false)
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
	scaleClientExpectedAction := []string{"patch", "get"}
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, stsgvr, false)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions := &ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, preconditions, namespace, name, count, nil, stsgvr, false)
	_, err = scaleFunc()
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
	scaleClientExpectedAction := []string{"patch"}
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClient := createFakeScaleClient("statefulsets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, stsgvr, false)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err == nil {
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
	scaler := NewScaler(scaleClient)
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil, stsgvr, false)
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
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 10, nil)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, nil, nil, nil, rsgvr, false)
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
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewConflict(api.Resource("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"patch", "get"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, rsgvr, false)
	pass, err := scaleFunc()
	if pass != false {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err != nil {
		t.Errorf("Did not expect an error on update failure, got %v", err)
	}
	preconditions := &ScalePrecondition{3, ""}
	scaleFunc = ScaleCondition(scaler, preconditions, namespace, name, count, nil, rsgvr, false)
	_, err = scaleFunc()
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
	verbsOnError := map[string]*apierrors.StatusError{
		"patch": apierrors.NewInvalid(api.Kind("Status"), "foo", nil),
	}
	scaleClientExpectedAction := []string{"patch"}
	scaleClient := createFakeScaleClient("replicasets", "foo", 2, verbsOnError)
	scaler := NewScaler(scaleClient)
	count := uint(3)
	name := "foo"
	namespace := "default"

	scaleFunc := ScaleCondition(scaler, nil, namespace, name, count, nil, rsgvr, false)
	pass, err := scaleFunc()
	if pass {
		t.Errorf("Expected an update failure to return pass = false, got pass = %v", pass)
	}
	if err == nil {
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
	scaler := NewScaler(scaleClient)
	preconditions := ScalePrecondition{2, ""}
	count := uint(3)
	name := "foo"
	err := scaler.Scale("default", name, count, &preconditions, nil, nil, rsgvr, false)
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
	scaleClient := createFakeScaleClient("deployments", "abc", 5, nil)
	// expected actions
	scaleClientExpectedAction := []string{"patch", "get", "update", "get", "update", "get", "get", "update", "get"}

	// test scenarios
	scenarios := []struct {
		name         string
		precondition *ScalePrecondition
		newSize      int
		targetGVR    schema.GroupVersionResource
		resName      string
		scaleGetter  scale.ScalesGetter
		expectError  bool
	}{
		// scenario 0: scale up the "abc" deployment without precondition
		{
			name:         "scale up the \"abc\" deployment without precondition",
			precondition: nil,
			newSize:      10,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 1: scale up the "abc" deployment
		{
			name:         "scale up the \"abc\" deployment",
			precondition: &ScalePrecondition{10, ""},
			newSize:      20,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 2: scale down the "abc" deployment
		{
			name:         "scale down the \"abs\" deployment",
			precondition: &ScalePrecondition{20, ""},
			newSize:      5,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 3: precondition error, expected size is 1,
		// note that the previous scenario (2) set the size to 5
		{
			name:         "precondition error, expected size is 1",
			precondition: &ScalePrecondition{1, ""},
			newSize:      5,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 4: precondition is not validated when the precondition size is set to -1
		{
			name:         "precondition is not validated when the size is set to -1",
			precondition: &ScalePrecondition{-1, ""},
			newSize:      5,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 5: precondition error, resource version mismatch
		{
			name:         "precondition error, resource version mismatch",
			precondition: &ScalePrecondition{5, "v1"},
			newSize:      5,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("running scenario %d: %s", index+1, scenario.name), func(t *testing.T) {
			target := NewScaler(scenario.scaleGetter)

			resVersion, err := target.ScaleSimple("default", scenario.resName, scenario.precondition, uint(scenario.newSize), scenario.targetGVR, false)

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

	// check actions
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

// TestGenericScale exercises GenericScaler.Scale method
func TestGenericScale(t *testing.T) {
	// test data
	scaleClient := createFakeScaleClient("deployments", "abc", 5, nil)
	// expected actions
	scaleClientExpectedAction := []string{"patch", "get", "update", "get", "get"}

	// test scenarios
	scenarios := []struct {
		name            string
		precondition    *ScalePrecondition
		newSize         int
		targetGVR       schema.GroupVersionResource
		resName         string
		scaleGetter     scale.ScalesGetter
		waitForReplicas *RetryParams
		expectError     bool
	}{
		// scenario 0: scale up the "abc" deployment without precondition
		{
			name:         "scale up the \"abc\" deployment without precondition",
			precondition: nil,
			newSize:      10,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		// scenario 1: scale up the "abc" deployment
		{
			name:         "scale up the \"abc\" deployment",
			precondition: &ScalePrecondition{10, ""},
			newSize:      20,
			targetGVR:    deploygvr,
			resName:      "abc",
			scaleGetter:  scaleClient,
		},
		//scenario 2: a resource name cannot be empty
		{
			name:         "a resource name cannot be empty",
			precondition: &ScalePrecondition{10, ""},
			newSize:      20,
			targetGVR:    deploygvr,
			resName:      "",
			scaleGetter:  scaleClient,
			expectError:  true,
		},
		// scenario 3: wait for replicas error due to status.Replicas != spec.Replicas
		{
			name:            "wait for replicas error due to status.Replicas != spec.Replicas",
			precondition:    &ScalePrecondition{10, ""},
			newSize:         20,
			targetGVR:       deploygvr,
			resName:         "abc",
			scaleGetter:     scaleClient,
			waitForReplicas: &RetryParams{time.Duration(5 * time.Second), time.Duration(5 * time.Second)},
			expectError:     true,
		},
	}

	// act
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			target := NewScaler(scenario.scaleGetter)

			err := target.Scale("default", scenario.resName, uint(scenario.newSize), scenario.precondition, nil, scenario.waitForReplicas, scenario.targetGVR, false)

			if scenario.expectError && err == nil {
				t.Fatal("expected an error but was not returned")
			}
			if !scenario.expectError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	// check actions
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

func createFakeScaleClient(resource string, resourceName string, replicas int, errorsOnVerb map[string]*apierrors.StatusError) *fakescale.FakeScaleClient {
	shouldReturnAnError := func(verb string) (*apierrors.StatusError, bool) {
		if anError, anErrorExists := errorsOnVerb[verb]; anErrorExists {
			return anError, true
		}
		return &apierrors.StatusError{}, false
	}
	newReplicas := int32(replicas)
	scaleClient := &fakescale.FakeScaleClient{}
	scaleClient.AddReactor("get", resource, func(rawAction testcore.Action) (handled bool, ret runtime.Object, err error) {
		action := rawAction.(testcore.GetAction)
		if action.GetName() != resourceName {
			return true, nil, fmt.Errorf("expected = %s, got = %s", resourceName, action.GetName())
		}
		if anError, should := shouldReturnAnError("get"); should {
			return true, nil, anError
		}
		obj := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      action.GetName(),
				Namespace: action.GetNamespace(),
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: newReplicas,
			},
		}
		return true, obj, nil
	})

	scaleClient.AddReactor("update", resource, func(rawAction testcore.Action) (handled bool, ret runtime.Object, err error) {
		action := rawAction.(testcore.UpdateAction)
		obj := action.GetObject().(*autoscalingv1.Scale)
		if obj.Name != resourceName {
			return true, nil, fmt.Errorf("expected = %s, got = %s", resourceName, obj.Name)
		}
		if anError, should := shouldReturnAnError("update"); should {
			return true, nil, anError
		}
		newReplicas = obj.Spec.Replicas
		return true, &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      obj.Name,
				Namespace: action.GetNamespace(),
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: newReplicas,
			},
		}, nil
	})

	scaleClient.AddReactor("patch", resource, func(rawAction testcore.Action) (handled bool, ret runtime.Object, err error) {
		action := rawAction.(testcore.PatchAction)
		pt := action.GetPatchType()
		if pt != types.MergePatchType {
			return true, nil, fmt.Errorf("unexpected patch type: expected = %s, got = %s", types.MergePatchType, pt)
		}
		var scale autoscalingv1.Scale
		err = json.Unmarshal(action.GetPatch(), &scale)
		if err != nil {
			return true, nil, fmt.Errorf("invalid patch: %s", err)
		}
		name := action.GetName()
		if name != resourceName {
			return true, nil, fmt.Errorf("expected = %s, got = %s", resourceName, name)
		}
		if anError, should := shouldReturnAnError("patch"); should {
			return true, nil, anError
		}
		newReplicas = scale.Spec.Replicas
		return true, &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: action.GetNamespace(),
			},
			Spec: autoscalingv1.ScaleSpec{
				Replicas: newReplicas,
			},
		}, nil
	})
	return scaleClient
}
