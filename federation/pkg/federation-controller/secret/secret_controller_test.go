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

package secret

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fake_fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	fake_kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
)

func TestSecretController(t *testing.T) {
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_fedclientset.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("secrets", &fakeClient.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	secretWatch := RegisterFakeWatch("secrets", &fakeClient.Fake)
	secretUpdateChan := RegisterFakeCopyOnUpdate("secrets", &fakeClient.Fake, secretWatch)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch("secrets", &cluster1Client.Fake)
	RegisterFakeList("secrets", &cluster1Client.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("secrets", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("secrets", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch("secrets", &cluster2Client.Fake)
	RegisterFakeList("secrets", &cluster2Client.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("secrets", &cluster2Client.Fake, cluster2Watch)

	secretController := NewSecretController(fakeClient)
	informerClientFactory := func(cluster *federation_api.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	setClientFactory(secretController.secretFederatedInformer, informerClientFactory)

	secretController.clusterAvailableDelay = time.Second
	secretController.secretReviewDelay = 50 * time.Millisecond
	secretController.smallDelay = 20 * time.Millisecond
	secretController.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	secretController.Run(stop)

	secret1 := api_v1.Secret{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "ns",
			SelfLink:  "/api/v1/namespaces/ns/secrets/test-secret",
		},
		Data: map[string][]byte{
			"A": []byte("ala ma kota"),
			"B": []byte("quick brown fox"),
		},
		Type: api_v1.SecretTypeOpaque,
	}

	// Test add federated secret.
	secretWatch.Add(&secret1)
	// There should be 2 updates to add both the finalizers.
	updatedSecret := GetSecretFromChan(secretUpdateChan)
	assert.True(t, secretController.hasFinalizerFunc(updatedSecret, deletionhelper.FinalizerDeleteFromUnderlyingClusters))
	updatedSecret = GetSecretFromChan(secretUpdateChan)
	assert.True(t, secretController.hasFinalizerFunc(updatedSecret, api_v1.FinalizerOrphan))
	secret1 = *updatedSecret

	// Verify that the secret is created in underlying cluster1.
	createdSecret := GetSecretFromChan(cluster1CreateChan)
	assert.NotNil(t, createdSecret)
	assert.Equal(t, secret1.Namespace, createdSecret.Namespace)
	assert.Equal(t, secret1.Name, createdSecret.Name)
	assert.True(t, secretsEqual(secret1, *createdSecret),
		fmt.Sprintf("expected: %v, actual: %v", secret1, *createdSecret))

	// Wait for the secret to appear in the informer store
	err := WaitForStoreUpdate(
		secretController.secretFederatedInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: secret1.Namespace, Name: secret1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "secret should have appeared in the informer store")

	checkAll := func(expected api_v1.Secret) CheckingFunction {
		return func(obj runtime.Object) error {
			glog.V(4).Infof("Checking %v", obj)
			s := obj.(*api_v1.Secret)
			if err := CompareObjectMeta(expected.ObjectMeta, s.ObjectMeta); err != nil {
				return err
			}
			if !reflect.DeepEqual(expected.Data, s.Data) {
				return fmt.Errorf("Data is different expected:%v actual:%v", expected.Data, s.Data)
			}
			if expected.Type != s.Type {
				return fmt.Errorf("Type is different expected:%v actual:%v", expected.Type, s.Type)
			}
			return nil
		}
	}

	// Test update federated secret.
	secret1.Annotations = map[string]string{
		"A": "B",
	}
	secretWatch.Modify(&secret1)
	err = CheckObjectFromChan(cluster1UpdateChan, checkAll(secret1))
	assert.NoError(t, err)

	// Wait for the secret to be updated in the informer store.
	err = WaitForSecretStoreUpdate(
		secretController.secretFederatedInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: secret1.Namespace, Name: secret1.Name}.String(),
		&secret1, wait.ForeverTestTimeout)
	assert.NoError(t, err, "secret should have been updated in the informer store")

	// Test update federated secret.
	secret1.Data = map[string][]byte{
		"config": []byte("myconfigurationfile"),
	}
	secretWatch.Modify(&secret1)
	err = CheckObjectFromChan(cluster1UpdateChan, checkAll(secret1))
	assert.NoError(t, err)

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdSecret2 := GetSecretFromChan(cluster2CreateChan)
	assert.NotNil(t, createdSecret2)
	assert.Equal(t, secret1.Name, createdSecret2.Name)
	assert.Equal(t, secret1.Namespace, createdSecret2.Namespace)
	assert.True(t, secretsEqual(secret1, *createdSecret2),
		fmt.Sprintf("expected: %v, actual: %v", secret1, *createdSecret2))

	close(stop)
}

func setClientFactory(informer util.FederatedInformer, informerClientFactory func(*federation_api.Cluster) (kubeclientset.Interface, error)) {
	testInformer := ToFederatedInformerForTestOnly(informer)
	testInformer.SetClientFactory(informerClientFactory)
}

func secretsEqual(a, b api_v1.Secret) bool {
	// Clear the SelfLink and ObjectMeta.Finalizers since they will be different
	// in resoure in federation control plane and resource in underlying cluster.
	a.SelfLink = ""
	b.SelfLink = ""
	a.ObjectMeta.Finalizers = []string{}
	b.ObjectMeta.Finalizers = []string{}
	return reflect.DeepEqual(a, b)
}

func GetSecretFromChan(c chan runtime.Object) *api_v1.Secret {
	secret := GetObjectFromChan(c).(*api_v1.Secret)
	return secret
}

// Wait till the store is updated with latest secret.
func WaitForSecretStoreUpdate(store util.FederatedReadOnlyStore, clusterName, key string, desiredSecret *api_v1.Secret, timeout time.Duration) error {
	retryInterval := 200 * time.Millisecond
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.GetByKey(clusterName, key)
		if !found || err != nil {
			glog.Infof("%s is not in the store", key)
			return false, err
		}
		equal := secretsEqual(*obj.(*api_v1.Secret), *desiredSecret)
		if !equal {
			glog.Infof("wrong content in the store expected:\n%v\nactual:\n%v\n", *desiredSecret, *obj.(*api_v1.Secret))
		}
		return equal, err
	})
	return err
}
