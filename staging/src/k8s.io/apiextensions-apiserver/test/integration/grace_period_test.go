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

package integration

import (
	"testing"

	"github.com/stretchr/testify/require"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGracePeriod(t *testing.T) {
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
	period := int64(30)
	uid := createdNoxuInstance.GetUID()
	err = noxuResourceClient.Delete(name, &metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &uid,
		},
		GracePeriodSeconds: &period,
	})
	require.NoError(t, err)
	// Testing Grace period is updating or not

	gottenNoxuInstance, err := noxuResourceClient.Get(name, metav1.GetOptions{})
	require.NoError(t, err)

	require.Equal(t, *gottenNoxuInstance.GetDeletionGracePeriodSeconds(), period)
}
