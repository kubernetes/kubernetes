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

package sync

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/stretchr/testify/assert"
)

const (
	horizontalpodautoscalers = "horizontalpodautoscalers"
	k8s1                     = "k8s-1"
	k8s2                     = "k8s-2"
	clusters                 = "clusters"
)

func TestAutoscalerController(t *testing.T) {

	cluster1 := NewCluster(k8s1, apiv1.ConditionTrue)
	cluster2 := NewCluster(k8s2, apiv1.ConditionTrue)

	fakeFedClient := &fakefedclientset.Clientset{}
	RegisterFakeList(clusters, &fakeFedClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1}})
	RegisterFakeList(horizontalpodautoscalers, &fakeFedClient.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	fedhpaWatch := RegisterFakeWatch(horizontalpodautoscalers, &fakeFedClient.Fake)
	hpaUpdateChan := RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &fakeFedClient.Fake, fedhpaWatch)
	clusterWatch := RegisterFakeWatch(clusters, &fakeFedClient.Fake)

	cluster1Client := &fakekubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch(horizontalpodautoscalers, &cluster1Client.Fake)
	RegisterFakeList(horizontalpodautoscalers, &cluster1Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate(horizontalpodautoscalers, &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fakekubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch(horizontalpodautoscalers, &cluster2Client.Fake)
	RegisterFakeList(horizontalpodautoscalers, &cluster2Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate(horizontalpodautoscalers, &cluster2Client.Fake, cluster2Watch)
	//cluster2UpdateChan :=  RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &cluster2Client.Fake, cluster2Watch)

	fedInformerClientFactory := func(cluster *federationapi.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case k8s1:
			return cluster1Client, nil
		case k8s2:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}

	hpaController := newFederationSyncController(fakeFedClient, federatedtypes.NewHpaAdapter(fakeFedClient))
	hpaController.minimizeLatency()

	hpaFedinformer := ToFederatedInformerForTestOnly(hpaController.informer)
	hpaFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	hpaController.Run(stopChan)

	hpa := newHpaWithReplicas("myautoscaler", newInt32(1), newInt32(70), 5)
	fedhpaWatch.Add(hpa)

	// 1: deletion finalizers are correctly added to hpa object
	updatedHpa := GetHpaFromChan(hpaUpdateChan)
	AssertHasFinalizer(t, updatedHpa, deletionhelper.FinalizerDeleteFromUnderlyingClusters)
	AssertHasFinalizer(t, updatedHpa, metav1.FinalizerOrphanDependents)

	// 2: the created hpa in federation gets created in registered cluster - cluster1
	hpa1 := GetHpaFromChan(cluster1CreateChan)
	assert.NotNil(t, hpa1)
	assert.Equal(t, hpa.Namespace, hpa1.Namespace)
	assert.Equal(t, hpa.Name, hpa1.Name)

	err := WaitForStoreUpdate(
		hpaController.informer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: hpa1.Namespace, Name: hpa1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "hpa from local cluster should have appeared in the informer store")

	// 3: Adding another cluster to federation gets an hpa in newly added cluster 2 also
	clusterWatch.Add(cluster2)
	hpa1 = GetHpaFromChan(cluster1UpdateChan)
	hpa2 := GetHpaFromChan(cluster2CreateChan)
	checkClusterConditions(t, hpa, hpa1, hpa2)

	// 4: update in the local hpa status gets reflected in federation object
	hpa1.Status.CurrentReplicas = hpa1.Spec.MaxReplicas - int32(3)
	hpa1.Status.DesiredReplicas = hpa1.Spec.MaxReplicas - int32(3)
	var lUtilisation int32
	lUtilisation = *hpa1.Spec.TargetCPUUtilizationPercentage - int32(20)
	hpa1.Status.CurrentCPUUtilizationPercentage = &lUtilisation
	cluster1Watch.Modify(hpa1)

	updatedHpa = GetHpaFromChan(hpaUpdateChan)
	assert.NotNil(t, updatedHpa)
	assert.Equal(t, hpa1.Status.CurrentReplicas, updatedHpa.Status.DesiredReplicas)
	assert.Equal(t, hpa1.Status.DesiredReplicas, updatedHpa.Status.CurrentReplicas)
	assert.Equal(t, (*updatedHpa.Spec.TargetCPUUtilizationPercentage - int32(20)),
		*updatedHpa.Status.CurrentCPUUtilizationPercentage)

	/* To Update
	// 5: Update of the federation object reflects in the local clusters
	hpa.Spec.MinReplicas = newInt32(4)
	hpa.Spec.MaxReplicas = 10
	hpa.Spec.TargetCPUUtilizationPercentage = newInt32(60)
	fedhpaWatch.Modify(hpa)
	hpa1 = GetHpaFromChan(cluster1UpdateChan)
	hpa2 = GetHpaFromChan(cluster2UpdateChan)
	checkClusterConditions(t, hpa, hpa1, hpa2)
	*/
}

func checkClusterConditions(t *testing.T, hpa *autoscalingv1.HorizontalPodAutoscaler, hpa1 *autoscalingv1.HorizontalPodAutoscaler, hpa2 *autoscalingv1.HorizontalPodAutoscaler) {
	assert.NotNil(t, hpa1)
	assert.NotNil(t, hpa2)
	assert.Equal(t, hpa.Namespace, hpa2.Namespace)
	assert.Equal(t, hpa.Name, hpa2.Name)
	// - Total of max matches the fed max
	assert.Equal(t, hpa.Spec.MaxReplicas, (hpa1.Spec.MaxReplicas + hpa2.Spec.MaxReplicas))
	// - Total of min is not less then fed min
	assert.Condition(t, func() bool {
		if *hpa.Spec.MinReplicas <= (*hpa1.Spec.MinReplicas + *hpa2.Spec.MinReplicas) {
			return true
		}
		return false
	})
}

func GetHpaFromChan(c chan runtime.Object) *autoscalingv1.HorizontalPodAutoscaler {
	if hpa := GetObjectFromChan(c); hpa == nil {
		return nil
	} else {
		return hpa.(*autoscalingv1.HorizontalPodAutoscaler)
	}
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}

func newHpaWithReplicas(name string, min, targetUtilisation *int32, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/mylink",
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: "HorizontalPodAutoscaler",
				Name: "myhpa",
			},
			MinReplicas:                    min,
			MaxReplicas:                    max,
			TargetCPUUtilizationPercentage: targetUtilisation,
		},
	}
}
