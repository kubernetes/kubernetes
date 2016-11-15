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
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"

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
	checkDeployment := func(base *extensionsv1.Deployment, replicas int32) CheckingFunction {
		return func(obj runtime.Object) error {
			if obj == nil {
				return fmt.Errorf("Observed object is nil")
			}
			d := obj.(*extensionsv1.Deployment)
			if err := CompareObjectMeta(base.ObjectMeta, d.ObjectMeta); err != nil {
				return err
			}
			if replicas != *d.Spec.Replicas {
				return fmt.Errorf("Replica count is different expected:%d observed:%d", replicas, *d.Spec.Replicas)
			}
			return nil
		}
	}
	assert.NoError(t, CheckObjectFromChan(cluster1CreateChan, checkDeployment(dep1, *dep1.Spec.Replicas)))
	err := WaitForStoreUpdate(
		deploymentController.fedDeploymentInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: dep1.Namespace, Name: dep1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "deployment should have appeared in the informer store")

	// Increase replica count. Expect to see the update in cluster1.
	newRep := int32(8)
	dep1.Spec.Replicas = &newRep
	deploymentsWatch.Modify(dep1)
	assert.NoError(t, CheckObjectFromChan(cluster1UpdateChan, checkDeployment(dep1, *dep1.Spec.Replicas)))

	// Add new cluster. Although rebalance = false, no pods have been created yet so it should
	// rebalance anyway.
	clusterWatch.Add(cluster2)
	assert.NoError(t, CheckObjectFromChan(cluster1UpdateChan, checkDeployment(dep1, *dep1.Spec.Replicas/2)))
	assert.NoError(t, CheckObjectFromChan(cluster2CreateChan, checkDeployment(dep1, *dep1.Spec.Replicas/2)))

	// Add new deployment with non-default replica placement preferences.
	dep2 := newDeploymentWithReplicas("deployment2", 9)
	dep2.Annotations = make(map[string]string)
	dep2.Annotations[FedDeploymentPreferencesAnnotation] = `{"rebalance": true,
		  "clusters": {
		    "cluster1": {"weight": 2},
		    "cluster2": {"weight": 1}
		}}`
	deploymentsWatch.Add(dep2)
	assert.NoError(t, CheckObjectFromChan(cluster1CreateChan, checkDeployment(dep2, 6)))
	assert.NoError(t, CheckObjectFromChan(cluster2CreateChan, checkDeployment(dep2, 3)))
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
