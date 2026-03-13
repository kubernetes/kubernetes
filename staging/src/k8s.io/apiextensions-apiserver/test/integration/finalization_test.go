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

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	testNamespace = "not-the-default"
	testFinalizer = "noxu.example.com/finalizer"
)

func TestFinalization(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	require.NoError(t, err)
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	require.NoError(t, err)

	ns := testNamespace
	name := "foo123"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	instance := fixtures.NewNoxuInstance(ns, name)
	instance.SetFinalizers([]string{testFinalizer})
	createdNoxuInstance, err := instantiateCustomResource(t, instance, noxuResourceClient, noxuDefinition)
	require.NoError(t, err)

	uid := createdNoxuInstance.GetUID()
	err = noxuResourceClient.Delete(context.TODO(), name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &uid,
		},
	})
	require.NoError(t, err)

	// Deleting something with a finalizer sets deletion timestamp to a not-nil value but does not
	// remove the object from the API server. Here we read it to confirm this.
	gottenNoxuInstance, err := noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
	require.NoError(t, err)

	require.NotNil(t, gottenNoxuInstance.GetDeletionTimestamp())

	// Trying to delete it again to confirm it will not remove the object because finalizer is still there.
	err = noxuResourceClient.Delete(context.TODO(), name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &uid,
		},
	})
	require.NoError(t, err)

	// Removing the finalizers to allow the following delete remove the object.
	// This step will fail if previous delete wrongly removed the object. The
	// object will be deleted as part of the finalizer update.
	for {
		gottenNoxuInstance.SetFinalizers(nil)
		_, err = noxuResourceClient.Update(context.TODO(), gottenNoxuInstance, metav1.UpdateOptions{})
		if err == nil {
			break
		}
		if !errors.IsConflict(err) {
			require.NoError(t, err) // Fail on unexpected error
		}
		gottenNoxuInstance, err = noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
		require.NoError(t, err)
	}

	// Check that the object is actually gone.
	_, err = noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
	require.Error(t, err)
	require.True(t, errors.IsNotFound(err), "%#v", err)
}

func TestFinalizationAndDeletion(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	require.NoError(t, err)
	defer tearDown()

	// Create a CRD.
	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	require.NoError(t, err)

	// Create a CR with a finalizer.
	ns := testNamespace
	name := "foo123"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	instance := fixtures.NewNoxuInstance(ns, name)
	instance.SetFinalizers([]string{testFinalizer})
	createdNoxuInstance, err := instantiateCustomResource(t, instance, noxuResourceClient, noxuDefinition)
	require.NoError(t, err)

	// Delete a CR. Because there's a finalizer, it will not get deleted now.
	uid := createdNoxuInstance.GetUID()
	err = noxuResourceClient.Delete(context.TODO(), name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &uid,
		},
	})
	require.NoError(t, err)

	// Check is the CR scheduled for deletion.
	gottenNoxuInstance, err := noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
	require.NoError(t, err)
	require.NotNil(t, gottenNoxuInstance.GetDeletionTimestamp())

	// Delete the CRD.
	fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)

	// Check is CR still there after the CRD deletion.
	gottenNoxuInstance, err = noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
	require.NoError(t, err)

	// Update the CR to remove the finalizer.
	for {
		gottenNoxuInstance.SetFinalizers(nil)
		_, err = noxuResourceClient.Update(context.TODO(), gottenNoxuInstance, metav1.UpdateOptions{})
		if err == nil {
			break
		}
		if !errors.IsConflict(err) {
			require.NoError(t, err) // Fail on unexpected error
		}
		gottenNoxuInstance, err = noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
		require.NoError(t, err)
	}

	// Verify the CR is gone.
	// It should return the NonFound error.
	_, err = noxuResourceClient.Get(context.TODO(), name, metav1.GetOptions{})
	if !errors.IsNotFound(err) {
		t.Fatalf("unable to delete cr: %v", err)
	}

	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		_, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), noxuDefinition.Name, metav1.GetOptions{})
		return errors.IsNotFound(err), err
	})
	if !errors.IsNotFound(err) {
		t.Fatalf("unable to delete crd: %v", err)
	}
}

func TestApplyCRDuringCRDFinalization(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	require.NoError(t, err)
	defer tearDown()

	// Create a CRD with a finalizer which will stall deletion
	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	noxuDefinition.SetFinalizers([]string{testFinalizer})
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	require.NoError(t, err)

	// Delete the CRD. Since it has a finalizer it will be stuck in terminating state
	err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Delete(t.Context(), noxuDefinition.Name, metav1.DeleteOptions{})
	require.NoError(t, err)

	// Wait for the CRD to have the Terminating condition set to True.
	// The handler checks IsCRDConditionTrue(crd, apiextensionsv1.Terminating) to block
	// CR creation, and this condition is set asynchronously by the CRD finalizer controller
	// after it observes the DeletionTimestamp. Without this wait, the Apply could succeed
	// if it races ahead of the controller setting the condition.
	err = wait.PollUntilContextTimeout(t.Context(), 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		crd, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, noxuDefinition.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return apiextensionshelpers.IsCRDConditionTrue(crd, apiextensionsv1.Terminating), nil
	})
	require.NoError(t, err, "timed out waiting for CRD Terminating condition to be set")

	// Try to create a CR using SSA. This should fail due to the CRD validation
	ns := testNamespace
	name := "foo123"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	instance := fixtures.NewNoxuInstance(ns, name)
	_, err = noxuResourceClient.Apply(t.Context(), name, instance, metav1.ApplyOptions{DryRun: []string{"All"}, FieldManager: "manager"})
	wantErr := `create not allowed while custom resource definition is terminating`
	require.ErrorContains(t, err, wantErr)
}
