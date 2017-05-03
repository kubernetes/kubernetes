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

package replicaset

import (
	"flag"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	core "k8s.io/client-go/testing"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	testutil "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/stretchr/testify/assert"
)

const (
	pods        = "pods"
	replicasets = "replicasets"
	k8s1        = "k8s-1"
	k8s2        = "k8s-2"
)

func TestParseFederationReplicaSetReference(t *testing.T) {
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

	rs := newReplicaSetWithReplicas("rs-1", 100)
	accessor, _ := meta.Accessor(rs)
	anno := accessor.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		accessor.SetAnnotations(anno)
	}
	for _, prefString := range successPrefs {
		anno[FedReplicaSetPreferencesAnnotation] = prefString
		pref, err := parseFederationReplicaSetReference(rs)
		assert.NotNil(t, pref)
		assert.Nil(t, err)
	}
	for _, prefString := range failedPrefes {
		anno[FedReplicaSetPreferencesAnnotation] = prefString
		pref, err := parseFederationReplicaSetReference(rs)
		assert.Nil(t, pref)
		assert.NotNil(t, err)
	}
}

func TestReplicaSetController(t *testing.T) {
	flag.Set("logtostderr", "true")
	flag.Set("v", "5")
	flag.Parse()

	replicaSetReviewDelay = 10 * time.Millisecond
	clusterAvailableDelay = 20 * time.Millisecond
	clusterUnavailableDelay = 60 * time.Millisecond
	allReplicaSetReviewDelay = 120 * time.Millisecond

	fedclientset := fedclientfake.NewSimpleClientset()
	fedrswatch := watch.NewFake()
	fedclientset.PrependWatchReactor(replicasets, core.DefaultWatchReactor(fedrswatch, nil))

	fedclientset.Federation().Clusters().Create(testutil.NewCluster(k8s1, apiv1.ConditionTrue))
	fedclientset.Federation().Clusters().Create(testutil.NewCluster(k8s2, apiv1.ConditionTrue))

	kube1clientset := kubeclientfake.NewSimpleClientset()
	kube1rswatch := watch.NewFake()
	kube1clientset.PrependWatchReactor(replicasets, core.DefaultWatchReactor(kube1rswatch, nil))
	kube1Podwatch := watch.NewFake()
	kube1clientset.PrependWatchReactor(pods, core.DefaultWatchReactor(kube1Podwatch, nil))
	kube2clientset := kubeclientfake.NewSimpleClientset()
	kube2rswatch := watch.NewFake()
	kube2clientset.PrependWatchReactor(replicasets, core.DefaultWatchReactor(kube2rswatch, nil))
	kube2Podwatch := watch.NewFake()
	kube2clientset.PrependWatchReactor(pods, core.DefaultWatchReactor(kube2Podwatch, nil))

	fedInformerClientFactory := func(cluster *fedv1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case k8s1:
			return kube1clientset, nil
		case k8s2:
			return kube2clientset, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}
	replicaSetController := NewReplicaSetController(fedclientset)
	rsFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedReplicaSetInformer)
	rsFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	go replicaSetController.Run(1, stopChan)

	rs := newReplicaSetWithReplicas("rs", 9)
	rs, _ = fedclientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Create(rs)
	fedrswatch.Add(rs)
	time.Sleep(1 * time.Second)

	rs1, _ := kube1clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	kube1rswatch.Add(rs1)
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1.Status.FullyLabeledReplicas = *rs1.Spec.Replicas
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs1.Status.AvailableReplicas = *rs1.Spec.Replicas
	rs1, _ = kube1clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs1)
	kube1rswatch.Modify(rs1)

	rs2, _ := kube2clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	kube2rswatch.Add(rs2)
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2.Status.FullyLabeledReplicas = *rs2.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas
	rs2.Status.AvailableReplicas = *rs2.Spec.Replicas
	rs2, _ = kube2clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs2)
	kube2rswatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedclientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)
	assert.Equal(t, rs.Status.FullyLabeledReplicas, rs1.Status.FullyLabeledReplicas+rs2.Status.FullyLabeledReplicas)
	assert.Equal(t, rs.Status.ReadyReplicas, rs1.Status.ReadyReplicas+rs2.Status.ReadyReplicas)
	assert.Equal(t, rs.Status.AvailableReplicas, rs1.Status.AvailableReplicas+rs2.Status.AvailableReplicas)

	var replicas int32 = 20
	rs.Spec.Replicas = &replicas
	rs, _ = fedclientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Update(rs)
	fedrswatch.Modify(rs)
	time.Sleep(1 * time.Second)

	rs1, _ = kube1clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1.Status.FullyLabeledReplicas = *rs1.Spec.Replicas
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs1.Status.AvailableReplicas = *rs1.Spec.Replicas
	rs1, _ = kube1clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs1)
	kube1rswatch.Modify(rs1)

	rs2, _ = kube2clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2.Status.FullyLabeledReplicas = *rs2.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas
	rs2.Status.AvailableReplicas = *rs2.Spec.Replicas
	rs2, _ = kube2clientset.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs2)
	kube2rswatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedclientset.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)
	assert.Equal(t, rs.Status.FullyLabeledReplicas, rs1.Status.FullyLabeledReplicas+rs2.Status.FullyLabeledReplicas)
	assert.Equal(t, rs.Status.ReadyReplicas, rs1.Status.ReadyReplicas+rs2.Status.ReadyReplicas)
	assert.Equal(t, rs.Status.AvailableReplicas, rs1.Status.AvailableReplicas+rs2.Status.AvailableReplicas)
}

func TestClusterReplicaState(t *testing.T) {
	uncalledPodsGetter := func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error) {
		t.Fatal("podsGetter should not be called when replica sets are all ready.")
		return nil, nil
	}

	podsByReplicaSet := make(map[*extensionsv1.ReplicaSet][]*apiv1.Pod)
	podsGetter := func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error) {
		pods, ok := podsByReplicaSet[replicaSet]
		if !ok {
			t.Fatalf("No pods found in test data for replica set named %v", replicaSet.Name)
			return nil, fmt.Errorf("Not found")
		}
		var podListPods []apiv1.Pod
		for _, pod := range pods {
			podListPods = append(podListPods, *pod)
		}
		return &apiv1.PodList{Items: podListPods}, nil
	}

	readyCondition := apiv1.PodCondition{Type: apiv1.PodReady}
	unschedulableCondition := apiv1.PodCondition{
		Type:               apiv1.PodScheduled,
		Status:             apiv1.ConditionFalse,
		Reason:             apiv1.PodReasonUnschedulable,
		LastTransitionTime: metav1.NewTime(time.Now().Add(-1 * time.Hour)),
	}

	one := int64(1)
	two := int64(2)

	tests := []struct {
		rs1Replicas                   int32
		rs2Replicas                   int32
		rs1ReadyReplicas              int32
		rs2ReadyReplicas              int32
		podsGetter                    func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error)
		pod1Phase                     apiv1.PodPhase
		pod1Condition                 apiv1.PodCondition
		pod2Phase                     apiv1.PodPhase
		pod2Condition                 apiv1.PodCondition
		cluster1Replicas              *int64
		cluster2Replicas              *int64
		cluster1UnschedulableReplicas *int64
		cluster2UnschedulableReplicas *int64
	}{
		{rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 2, rs2ReadyReplicas: 2, podsGetter: uncalledPodsGetter, cluster1Replicas: &two, cluster2Replicas: &two},
		{rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 1, rs2ReadyReplicas: 2, podsGetter: podsGetter, pod1Phase: apiv1.PodRunning, pod1Condition: readyCondition, pod2Phase: apiv1.PodPending, cluster1Replicas: &one, cluster2Replicas: &two},
		{rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 1, rs2ReadyReplicas: 2, podsGetter: podsGetter, pod1Phase: apiv1.PodRunning, pod1Condition: readyCondition, pod2Phase: apiv1.PodPending, pod2Condition: unschedulableCondition, cluster1Replicas: &one, cluster2Replicas: &two, cluster1UnschedulableReplicas: &one},
	}

	for _, tt := range tests {
		clusters := []string{"one", "two"}
		replicaSetsByCluster := make(map[string]*extensionsv1.ReplicaSet)
		replicaSetGetter := func(clusterName string, key string) (interface{}, bool, error) {
			rs, ok := replicaSetsByCluster[clusterName]
			if !ok {
				t.Fatalf("No replica set found in test data for %v", clusterName)
				return nil, false, fmt.Errorf("Not found")
			}
			return rs, true, nil
		}
		rs1 := newReplicaSetWithReplicas("one", tt.rs1Replicas)
		rs2 := newReplicaSetWithReplicas("two", tt.rs2Replicas)
		rs1.Spec.Replicas = &tt.rs1Replicas
		rs2.Spec.Replicas = &tt.rs2Replicas
		rs1.Status.ReadyReplicas = tt.rs1ReadyReplicas
		rs2.Status.ReadyReplicas = tt.rs2ReadyReplicas

		replicaSetsByCluster["one"] = rs1
		replicaSetsByCluster["two"] = rs2

		pod1 := newPod("one")
		pod2 := newPod("two")
		podThree := newPod("three")
		podFour := newPod("four")

		pod1.Status.Phase = tt.pod1Phase
		pod2.Status.Phase = tt.pod2Phase
		pod1.Status.Conditions = []apiv1.PodCondition{tt.pod1Condition}
		pod2.Status.Conditions = []apiv1.PodCondition{tt.pod2Condition}

		podsByReplicaSet[rs1] = []*apiv1.Pod{pod1, pod2}
		podsByReplicaSet[rs2] = []*apiv1.Pod{podThree, podFour}

		current, estimatedCapacity, err := clustersReplicaState(clusters, "", replicaSetGetter, tt.podsGetter)

		assert.Nil(t, err)

		wantedCurrent := make(map[string]int64)
		if tt.cluster1Replicas != nil {
			wantedCurrent["one"] = *tt.cluster1Replicas
		}
		if tt.cluster2Replicas != nil {
			wantedCurrent["two"] = *tt.cluster2Replicas
		}
		assert.Equal(t, wantedCurrent, current)

		wantedEstimatedCapacity := make(map[string]int64)
		if tt.cluster1UnschedulableReplicas != nil {
			wantedEstimatedCapacity["one"] = *tt.cluster1UnschedulableReplicas
		}
		if tt.cluster2UnschedulableReplicas != nil {
			wantedEstimatedCapacity["two"] = *tt.cluster2UnschedulableReplicas
		}
		assert.Equal(t, wantedEstimatedCapacity, estimatedCapacity)
	}
}

func newReplicaSetWithReplicas(name string, replicas int32) *extensionsv1.ReplicaSet {
	return &extensionsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/replicasets/name",
		},
		Spec: extensionsv1.ReplicaSetSpec{
			Replicas: &replicas,
		},
	}
}

func newPod(name string) *apiv1.Pod {
	return &apiv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
	}
}
