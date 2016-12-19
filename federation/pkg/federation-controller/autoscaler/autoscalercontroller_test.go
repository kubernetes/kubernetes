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

package autoscaler

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"

	"github.com/golang/glog"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/stretchr/testify/assert"
)

func TestAutoscalerController(t *testing.T) {

	cluster1 := NewCluster("cluster1", apiv1.ConditionTrue)
	cluster2 := NewCluster("cluster2", apiv1.ConditionTrue)

	hpa1 := autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: apiv1.ObjectMeta{
			Name:      "myautoscaler",
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/mylink",
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: "ReplicationController",
				Name: "myrc",
			},
			MinReplicas:                    newInt32(1),
			MaxReplicas:                    5,
			TargetCPUUtilizationPercentage: newInt32(70),
		},
	}

	fakeClient := &fakefedclientset.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1}})
	RegisterFakeList("horizontalpodautoscalers", &fakeClient.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})

	hpaWatch := RegisterFakeWatch("horizontalpodautoscalers", &fakeClient.Fake)
	hpaUpdateChan := RegisterFakeCopyOnUpdate("horizontalpodautoscalers", &fakeClient.Fake, hpaWatch)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fakekubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch("horizontalpodautoscalers", &cluster1Client.Fake)
	RegisterFakeList("horizontalpodautoscalers", &cluster1Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("horizontalpodautoscalers", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("horizontalpodautoscalers", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fakekubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch("horizontalpodautoscalers", &cluster2Client.Fake)
	RegisterFakeList("horizontalpodautoscalers", &cluster2Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("horizontalpodautoscalers", &cluster2Client.Fake, cluster2Watch)

	hpaController := NewAutoscalerController(fakeClient)
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
	setClientFactory(hpaController.fedAutoscalerInformer, informerClientFactory)

	stop := make(chan struct{})
	hpaController.Run(stop)

	// Test add federated hpa.
	hpaWatch.Add(&hpa1)
	// There should be 2 updates to add both the finalizers.
	updatedHpa := GetHpaFromChan(hpaUpdateChan)
	assert.True(t, hpaController.hasFinalizerFunc(updatedHpa, deletionhelper.FinalizerDeleteFromUnderlyingClusters))
	updatedHpa = GetHpaFromChan(hpaUpdateChan)
	assert.True(t, hpaController.hasFinalizerFunc(updatedHpa, apiv1.FinalizerOrphan))
	hpa1 = *updatedHpa

	// Verify that the hpa is created in underlying cluster1.
	createdHpa := GetHpaFromChan(cluster1CreateChan)
	assert.NotNil(t, createdHpa)
	assert.Equal(t, hpa1.Namespace, createdHpa.Namespace)
	assert.Equal(t, hpa1.Name, createdHpa.Name)
	assert.True(t, hpasEqual(hpa1, *createdHpa),
		fmt.Sprintf("expected: %v, actual: %v", hpa1, *createdHpa))

	// Wait for the hpa to appear in the informer store
	err := WaitForStoreUpdate(
		hpaController.fedAutoscalerInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: hpa1.Namespace, Name: hpa1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "hpa should have appeared in the informer store")

	checkAll := func(expected autoscalingv1.HorizontalPodAutoscaler) CheckingFunction {
		return func(obj runtime.Object) error {
			glog.V(4).Infof("Checking %v", obj)
			s := obj.(*autoscalingv1.HorizontalPodAutoscaler)
			if err := CompareObjectMeta(expected.ObjectMeta, s.ObjectMeta); err != nil {
				return err
			}
			if !reflect.DeepEqual(expected.Spec, s.Spec) {
				return fmt.Errorf("Data is different expected:%v actual:%v", expected.Spec, s.Spec)
			}
			return nil
		}
	}

	// Test update federated hpa.
	hpa1.Annotations = map[string]string{
		"A": "B",
	}
	hpaWatch.Modify(&hpa1)
	err = CheckObjectFromChan(cluster1UpdateChan, checkAll(hpa1))
	assert.NoError(t, err)

	// Wait for the hpa to be updated in the informer store.
	err = WaitForHpaStoreUpdate(
		hpaController.fedAutoscalerInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: hpa1.Namespace, Name: hpa1.Name}.String(),
		&hpa1, wait.ForeverTestTimeout)
	assert.NoError(t, err, "hpa should have been updated in the informer store")

	// Test update federated hpa.
	hpa1.Spec = autoscalingv1.HorizontalPodAutoscalerSpec{
		ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
			Kind: "ReplicationController",
			Name: "anotherrc",
		},
	}

	hpaWatch.Modify(&hpa1)
	err = CheckObjectFromChan(cluster1UpdateChan, checkAll(hpa1))
	assert.NoError(t, err)

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdHpa2 := GetHpaFromChan(cluster2CreateChan)
	assert.NotNil(t, createdHpa2)
	assert.Equal(t, hpa1.Name, createdHpa2.Name)
	assert.Equal(t, hpa1.Namespace, createdHpa2.Namespace)
	assert.True(t, hpasEqual(hpa1, *createdHpa2),
		fmt.Sprintf("expected: %v, actual: %v", hpa1, *createdHpa2))

	close(stop)

}

func setClientFactory(informer util.FederatedInformer, informerClientFactory func(*federationapi.Cluster) (kubeclientset.Interface, error)) {
	testInformer := ToFederatedInformerForTestOnly(informer)
	testInformer.SetClientFactory(informerClientFactory)
}

func WaitForHpaStoreUpdate(store util.FederatedReadOnlyStore, clusterName, key string, desiredHpa *autoscalingv1.HorizontalPodAutoscaler, timeout time.Duration) error {
	retryInterval := 200 * time.Millisecond
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.GetByKey(clusterName, key)
		if !found || err != nil {
			glog.Infof("%s is not in the store", key)
			return false, err
		}
		equal := hpasEqual(*obj.(*autoscalingv1.HorizontalPodAutoscaler), *desiredHpa)
		if !equal {
			glog.Infof("wrong content in the store expected:\n%v\nactual:\n%v\n", *desiredHpa, *obj.(*autoscalingv1.HorizontalPodAutoscaler))
		}
		return equal, err
	})
	return err
}

func GetHpaFromChan(c chan runtime.Object) *autoscalingv1.HorizontalPodAutoscaler {
	hpa := GetObjectFromChan(c).(*autoscalingv1.HorizontalPodAutoscaler)
	return hpa
}

func hpasEqual(a, b autoscalingv1.HorizontalPodAutoscaler) bool {
	// Clear the SelfLink and ObjectMeta.Finalizers since they will be different
	// in resoure in federation control plane and resource in underlying cluster.
	a.SelfLink = ""
	b.SelfLink = ""
	a.ObjectMeta.Finalizers = []string{}
	b.ObjectMeta.Finalizers = []string{}
	return reflect.DeepEqual(a, b)
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
