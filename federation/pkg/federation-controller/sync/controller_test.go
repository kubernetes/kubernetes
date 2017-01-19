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

package sync

import (
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/stretchr/testify/assert"
)

const (
	secrets  string = "secrets"
	clusters string = "clusters"
)

type testData struct {
	obj        runtime.Object
	name       string
	operations []util.FederatedOperation
}

func NewSecretObjectAnnotations(namespace, name string, annotations map[string]string) runtime.Object {
	return &apiv1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			SelfLink:    "/api/v1/namespaces/" + namespace + "/secrets/" + name,
			Annotations: annotations,
		},
		Data: map[string][]byte{
			"A": []byte("ala ma kota"),
		},
		Type: apiv1.SecretTypeOpaque,
	}
}

func TestGetClusterOperations(t *testing.T) {
	cluster1 := NewCluster("cluster1", apiv1.ConditionTrue)
	cluster2 := NewCluster("cluster2", apiv1.ConditionTrue)

	cluster1.ObjectMeta.Labels["location"] = "europe"
	cluster1.ObjectMeta.Labels["environment"] = "prod"
	cluster1.ObjectMeta.Labels["version"] = "15"
	cluster2.ObjectMeta.Labels["location"] = "europe"
	cluster2.ObjectMeta.Labels["environment"] = "test"
	cluster2.ObjectMeta.Labels["version"] = "16"

	fakeClient := &fakefedclientset.Clientset{}
	RegisterFakeList(clusters, &fakeClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1, *cluster2}})
	RegisterFakeList(secrets, &fakeClient.Fake, &apiv1.SecretList{Items: []apiv1.Secret{}})
	secretWatch := RegisterFakeWatch(secrets, &fakeClient.Fake)
	RegisterFakeCopyOnUpdate(secrets, &fakeClient.Fake, secretWatch)

	cluster1Client := &fakekubeclientset.Clientset{}
	cluster2Client := &fakekubeclientset.Clientset{}

	s := newFederationSyncController(fakeClient, federatedtypes.NewSecretAdapter(fakeClient))

	informerClientFactory := func(cluster *federationapi.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	setClientFactory(s.informer, informerClientFactory)

	stop := make(chan struct{})

	s.Run(stop)

	tests := []testData{}

	//test1 secret sent to all clusters
	secret := NewSecretObjectAnnotations("ns", "secret1", map[string]string{})
	operations := make([]util.FederatedOperation, 0)
	operations = append(operations, util.FederatedOperation{
		Type:        util.OperationTypeAdd,
		Obj:         secret,
		ClusterName: "cluster1",
	})
	operations = append(operations, util.FederatedOperation{
		Type:        util.OperationTypeAdd,
		Obj:         secret,
		ClusterName: "cluster2",
	})
	tests = append(tests, testData{obj: secret, name: "secret1 to all clusters", operations: operations})

	//test2 secret sent to only prod clusters in europe
	secret = NewSecretObjectAnnotations("ns", "secret2", map[string]string{
		federationapi.FederationClusterSelector: "[{\"key\": \"location\", \"operator\": \"=\", \"values\": [\"europe\"]},{\"key\": \"environment\", \"operator\": \"=\", \"values\": [\"prod\"]}]",
	})
	operations = make([]util.FederatedOperation, 0)
	operations = append(operations, util.FederatedOperation{
		Type:        util.OperationTypeAdd,
		Obj:         secret,
		ClusterName: "cluster1",
	})
	tests = append(tests, testData{obj: secret, name: "secret2 to prod clusters in europe", operations: operations})

	//test3 secret not matching any cluster
	secret = NewSecretObjectAnnotations("ns", "secret3", map[string]string{
		federationapi.FederationClusterSelector: "[{\"key\": \"federated\", \"operator\": \"=\", \"values\": [\"false\"]},{\"key\": \"id\", \"operator\": \"=\", \"values\": [\"2e0f30ea-a9de-4817-84a7-0b48a0e881e0\"]}]",
	})
	operations = make([]util.FederatedOperation, 0)
	tests = append(tests, testData{obj: secret, name: "secret3 only exists on federated controller", operations: operations})

	//test4 numeric greater than
	secret = NewSecretObjectAnnotations("ns", "secret4", map[string]string{
		federationapi.FederationClusterSelector: "[{\"key\": \"version\", \"operator\": \"gt\", \"values\": [\"15\"]}]",
	})
	operations = make([]util.FederatedOperation, 0)
	operations = append(operations, util.FederatedOperation{
		Type:        util.OperationTypeAdd,
		Obj:         secret,
		ClusterName: "cluster2",
	})
	tests = append(tests, testData{obj: secret, name: "secret4 version greater than", operations: operations})

	for _, test := range tests {
		secretWatch.Add(test.obj)
		//Make sure the federated object is synced
		for !s.isSynced() {
			s.deliver(s.adapter.NamespacedName(test.obj), s.clusterAvailableDelay, false)
			time.Sleep(100 * time.Millisecond)
		}
		//Make sure it exists in cache and is the version we expect
		for {
			_, exist, _ := s.store.GetByKey(s.adapter.NamespacedName(test.obj).String())
			if exist {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}

		err, operationResults := s.getClusterOperations(test.obj)
		assert.Nil(t, err)
		assert.Equal(t, len(operationResults), len(test.operations))

		for _, operation := range test.operations {
			for _, result := range operationResults {
				if result.ClusterName == operation.ClusterName {
					fmt.Printf("Test: %s Cluster: %s Expected: %s Result: %s\n", test.name, operation.ClusterName, operation.Type, result.Type)
					assert.Equal(t, operation.Type, result.Type)
				}
			}
		}
	}

	close(stop)
}
