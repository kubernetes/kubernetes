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

package deployment

import (
	"flag"
	"fmt"
	"testing"
	"time"

	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fake_fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/fake"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/meta"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	fake_kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestParseFederationDeploymentPreference(t *testing.T) {
	successPrefs := []string{
		`{"rebalance": true,
		  "clusters": {
		    "k8s-1": {"minReplicas": 10, "maxReplicas": 20, "weight": 2},
		    "*": {"weight": 1}
		}}`,
	}
	failedPrefes := []string{
		`{`, // bad json
	}

	rs := newDeploymentWithReplicas("d-1", 100)
	accessor, _ := meta.Accessor(rs)
	anno := accessor.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		accessor.SetAnnotations(anno)
	}
	for _, prefString := range successPrefs {
		anno[FedDeploymentPreferencesAnnotation] = prefString
		pref, err := parseFederationDeploymentPreference(rs)
		assert.NotNil(t, pref)
		assert.Nil(t, err)
	}
	for _, prefString := range failedPrefes {
		anno[FedDeploymentPreferencesAnnotation] = prefString
		pref, err := parseFederationDeploymentPreference(rs)
		assert.Nil(t, pref)
		assert.NotNil(t, err)
	}
}

func TestDeploymentController(t *testing.T) {
	flag.Set("logtostderr", "true")
	flag.Set("v", "5")
	flag.Parse()

	deploymentReviewDelay = 500 * time.Millisecond
	clusterAvailableDelay = 100 * time.Millisecond
	clusterUnavailableDelay = 100 * time.Millisecond
	allDeploymentReviewDelay = 500 * time.Millisecond

	cluster1 := NewCluster("cluster1", apiv1.ConditionTrue)
	cluster2 := NewCluster("cluster2", apiv1.ConditionTrue)

	fakeClient := &fake_fedclientset.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &fedv1.ClusterList{Items: []fedv1.Cluster{*cluster1}})
	deploymentsWatch := RegisterFakeWatch("deployments", &fakeClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch("deployments", &cluster1Client.Fake)
	_ = RegisterFakeWatch("pods", &cluster1Client.Fake)
	RegisterFakeList("deployments", &cluster1Client.Fake, &extensionsv1.DeploymentList{Items: []extensionsv1.Deployment{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("deployments", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("deployments", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch("deployments", &cluster2Client.Fake)
	_ = RegisterFakeWatch("pods", &cluster2Client.Fake)
	RegisterFakeList("deployments", &cluster2Client.Fake, &extensionsv1.DeploymentList{Items: []extensionsv1.Deployment{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("deployments", &cluster2Client.Fake, cluster2Watch)

	deploymentController := NewDeploymentController(fakeClient)
	clientFactory := func(cluster *fedv1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	ToFederatedInformerForTestOnly(deploymentController.fedDeploymentInformer).SetClientFactory(clientFactory)
	ToFederatedInformerForTestOnly(deploymentController.fedPodInformer).SetClientFactory(clientFactory)

	stop := make(chan struct{})
	go deploymentController.Run(5, stop)

	// Create deployment. Expect to see it in cluster1.
	dep1 := newDeploymentWithReplicas("depA", 6)
	deploymentsWatch.Add(dep1)
	createdDep1 := GetDeploymentFromChan(cluster1CreateChan)
	assert.NotNil(t, createdDep1)
	assert.Equal(t, dep1.Namespace, createdDep1.Namespace)
	assert.Equal(t, dep1.Name, createdDep1.Name)
	assert.Equal(t, dep1.Spec.Replicas, createdDep1.Spec.Replicas)

	// Increase replica count. Expect to see the update in cluster1.
	newRep := int32(8)
	dep1.Spec.Replicas = &newRep
	deploymentsWatch.Modify(dep1)
	updatedDep1 := GetDeploymentFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedDep1)
	assert.Equal(t, dep1.Namespace, updatedDep1.Namespace)
	assert.Equal(t, dep1.Name, updatedDep1.Name)
	assert.Equal(t, dep1.Spec.Replicas, updatedDep1.Spec.Replicas)

	// Add new cluster. Although rebalance = false, no pods have been created yet so it should
	// rebalance anyway.
	clusterWatch.Add(cluster2)
	updatedDep1 = GetDeploymentFromChan(cluster1UpdateChan)
	createdDep2 := GetDeploymentFromChan(cluster2CreateChan)
	assert.NotNil(t, updatedDep1)
	assert.NotNil(t, createdDep2)

	assert.Equal(t, dep1.Namespace, createdDep2.Namespace)
	assert.Equal(t, dep1.Name, createdDep2.Name)
	assert.Equal(t, *dep1.Spec.Replicas/2, *createdDep2.Spec.Replicas)
	assert.Equal(t, *dep1.Spec.Replicas/2, *updatedDep1.Spec.Replicas)
}

func GetDeploymentFromChan(c chan runtime.Object) *extensionsv1.Deployment {
	secret := GetObjectFromChan(c).(*extensionsv1.Deployment)
	return secret
}

func newDeploymentWithReplicas(name string, replicas int32) *extensionsv1.Deployment {
	return &extensionsv1.Deployment{
		ObjectMeta: apiv1.ObjectMeta{
			Name:      name,
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/deployments/name",
		},
		Spec: extensionsv1.DeploymentSpec{
			Replicas: &replicas,
		},
	}
}
