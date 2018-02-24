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
	"testing"

	"github.com/stretchr/testify/require"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFinalization(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	require.NoError(t, err)
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	require.NoError(t, err)

	ns := "not-the-default"
	name := "foo123"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)

	instance := testserver.NewNoxuInstance(ns, name)
	instance.SetFinalizers([]string{"noxu.example.com/finalizer"})
	createdNoxuInstance, err := instantiateCustomResource(t, instance, noxuResourceClient, noxuDefinition)
	require.NoError(t, err)

	uid := createdNoxuInstance.GetUID()
	err = noxuResourceClient.Delete(name, &metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &uid,
		},
	})
	require.NoError(t, err)

	// Deleting something with a finalizer sets deletion timestamp to a not-nil value but does not
	// remove the object from the API server. Here we read it to confirm this.
	gottenNoxuInstance, err := noxuResourceClient.Get(name, metav1.GetOptions{})
	require.NoError(t, err)

	require.NotNil(t, gottenNoxuInstance.GetDeletionTimestamp())

	// Trying to delete it again to confirm it will not remove the object because finalizer is still there.
	err = noxuResourceClient.Delete(name, &metav1.DeleteOptions{
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
		_, err = noxuResourceClient.Update(gottenNoxuInstance)
		if err == nil {
			break
		}
		if !errors.IsConflict(err) {
			require.NoError(t, err) // Fail on unexpected error
		}
		gottenNoxuInstance, err = noxuResourceClient.Get(name, metav1.GetOptions{})
		require.NoError(t, err)
	}

	// Check that the object is actually gone.
	_, err = noxuResourceClient.Get(name, metav1.GetOptions{})
	require.Error(t, err)
	require.True(t, errors.IsNotFound(err), "%#v", err)
}
