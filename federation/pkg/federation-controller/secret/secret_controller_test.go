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
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestSecretController(t *testing.T) {
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("secrets", &fakeClient.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	secretWatch := RegisterFakeWatch("secrets", &fakeClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	cluster1Watch := RegisterFakeWatch("secrets", &cluster1Client.Fake)
	RegisterFakeList("secrets", &cluster1Client.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("secrets", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("secrets", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	cluster2Watch := RegisterFakeWatch("secrets", &cluster2Client.Fake)
	RegisterFakeList("secrets", &cluster2Client.Fake, &api_v1.SecretList{Items: []api_v1.Secret{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("secrets", &cluster2Client.Fake, cluster2Watch)

	secretController := NewSecretController(fakeClient)
	informer := ToFederatedInformerForTestOnly(secretController.secretFederatedInformer)
	informer.SetClientFactory(func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	})

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
	createdSecret := GetSecretFromChan(cluster1CreateChan)
	assert.NotNil(t, createdSecret)
	assert.Equal(t, secret1.Namespace, createdSecret.Namespace)
	assert.Equal(t, secret1.Name, createdSecret.Name)
	assert.True(t, secretsEqual(secret1, *createdSecret))

	// Test update federated secret.
	secret1.Annotations = map[string]string{
		"A": "B",
	}
	secretWatch.Modify(&secret1)
	updatedSecret := GetSecretFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedSecret)
	assert.Equal(t, secret1.Name, updatedSecret.Name)
	assert.Equal(t, secret1.Namespace, updatedSecret.Namespace)
	assert.True(t, secretsEqual(secret1, *updatedSecret))

	// Test update federated secret.
	secret1.Data = map[string][]byte{
		"config": []byte("myconfigurationfile"),
	}
	secretWatch.Modify(&secret1)
	updatedSecret2 := GetSecretFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedSecret)
	assert.Equal(t, secret1.Name, updatedSecret.Name)
	assert.Equal(t, secret1.Namespace, updatedSecret.Namespace)
	assert.True(t, secretsEqual(secret1, *updatedSecret2))

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdSecret2 := GetSecretFromChan(cluster2CreateChan)
	assert.NotNil(t, createdSecret2)
	assert.Equal(t, secret1.Name, createdSecret2.Name)
	assert.Equal(t, secret1.Namespace, createdSecret2.Namespace)
	assert.True(t, secretsEqual(secret1, *createdSecret2))

	close(stop)
}

func secretsEqual(a, b api_v1.Secret) bool {
	a.SelfLink = ""
	b.SelfLink = ""
	return reflect.DeepEqual(a, b)
}

func GetSecretFromChan(c chan runtime.Object) *api_v1.Secret {
	secret := GetObjectFromChan(c).(*api_v1.Secret)
	return secret
}
