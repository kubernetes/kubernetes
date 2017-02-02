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
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	core "k8s.io/client-go/testing"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/stretchr/testify/assert"
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
	replicaSetReviewDelay = 10 * time.Millisecond
	clusterAvailableDelay = 20 * time.Millisecond
	clusterUnavailableDelay = 60 * time.Millisecond
	allReplicaSetReviewDelay = 120 * time.Millisecond

	fedClientSet := fedclientfake.NewSimpleClientset()
	fedRsWatch := watch.NewFake()
	fedClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(fedRsWatch, nil))

	fedClientSet.Federation().Clusters().Create(testutil.NewCluster("k8s-1", apiv1.ConditionTrue))
	fedClientSet.Federation().Clusters().Create(testutil.NewCluster("k8s-2", apiv1.ConditionTrue))

	kube1ClientSet := kubeclientfake.NewSimpleClientset()
	kube1RsWatch := watch.NewFake()
	kube1ClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube1RsWatch, nil))
	kube1PodWatch := watch.NewFake()
	kube1ClientSet.PrependWatchReactor("pods", core.DefaultWatchReactor(kube1PodWatch, nil))
	kube2ClientSet := kubeclientfake.NewSimpleClientset()
	kube2RsWatch := watch.NewFake()
	kube2ClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube2RsWatch, nil))
	kube2PodWatch := watch.NewFake()
	kube2ClientSet.PrependWatchReactor("pods", core.DefaultWatchReactor(kube2PodWatch, nil))

	fedInformerClientFactory := func(cluster *fedv1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case "k8s-1":
			return kube1ClientSet, nil
		case "k8s-2":
			return kube2ClientSet, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}
	replicaSetController := NewReplicaSetController(fedClientSet)
	rsFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedReplicaSetInformer)
	rsFedinformer.SetClientFactory(fedInformerClientFactory)
	podFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedPodInformer)
	podFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	go replicaSetController.Run(1, stopChan)

	rs := newReplicaSetWithReplicas("rs", 9)
	rs, _ = fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Create(rs)
	fedRsWatch.Add(rs)
	time.Sleep(1 * time.Second)

	rs1, _ := kube1ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	kube1RsWatch.Add(rs1)
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1.Status.FullyLabeledReplicas = *rs1.Spec.Replicas
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs1.Status.AvailableReplicas = *rs1.Spec.Replicas
	rs1, _ = kube1ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs1)
	kube1RsWatch.Modify(rs1)

	rs2, _ := kube2ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	kube2RsWatch.Add(rs2)
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2.Status.FullyLabeledReplicas = *rs2.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas
	rs2.Status.AvailableReplicas = *rs2.Spec.Replicas
	rs2, _ = kube2ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs2)
	kube2RsWatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)
	assert.Equal(t, rs.Status.FullyLabeledReplicas, rs1.Status.FullyLabeledReplicas+rs2.Status.FullyLabeledReplicas)
	assert.Equal(t, rs.Status.ReadyReplicas, rs1.Status.ReadyReplicas+rs2.Status.ReadyReplicas)
	assert.Equal(t, rs.Status.AvailableReplicas, rs1.Status.AvailableReplicas+rs2.Status.AvailableReplicas)

	var replicas int32 = 20
	rs.Spec.Replicas = &replicas
	rs, _ = fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Update(rs)
	fedRsWatch.Modify(rs)
	time.Sleep(1 * time.Second)

	rs1, _ = kube1ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1.Status.FullyLabeledReplicas = *rs1.Spec.Replicas
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs1.Status.AvailableReplicas = *rs1.Spec.Replicas
	rs1, _ = kube1ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs1)
	kube1RsWatch.Modify(rs1)

	rs2, _ = kube2ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2.Status.FullyLabeledReplicas = *rs2.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas
	rs2.Status.AvailableReplicas = *rs2.Spec.Replicas
	rs2, _ = kube2ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).UpdateStatus(rs2)
	kube2RsWatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)
	assert.Equal(t, rs.Status.FullyLabeledReplicas, rs1.Status.FullyLabeledReplicas+rs2.Status.FullyLabeledReplicas)
	assert.Equal(t, rs.Status.ReadyReplicas, rs1.Status.ReadyReplicas+rs2.Status.ReadyReplicas)
	assert.Equal(t, rs.Status.AvailableReplicas, rs1.Status.AvailableReplicas+rs2.Status.AvailableReplicas)
}

// The tests adds a puppet federated RS and 2 local RS and hopes that the puppet
// status and replicas are correctly updated.
func TestReplicaSetControllerPuppets(t *testing.T) {

	replicaSetReviewDelay = 10 * time.Millisecond
	clusterAvailableDelay = 20 * time.Millisecond
	clusterUnavailableDelay = 60 * time.Millisecond
	allReplicaSetReviewDelay = 120 * time.Millisecond

	fedClientSet := fedclientfake.NewSimpleClientset()
	fedRsWatch := watch.NewFake()
	fedClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(fedRsWatch, nil))

	fedClientSet.Federation().Clusters().Create(testutil.NewCluster("k8s-1", apiv1.ConditionTrue))
	fedClientSet.Federation().Clusters().Create(testutil.NewCluster("k8s-2", apiv1.ConditionTrue))

	kube1ClientSet := kubeclientfake.NewSimpleClientset()
	kube1RsWatch := watch.NewFake()
	kube1ClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube1RsWatch, nil))
	kube1PodWatch := watch.NewFake()
	kube1ClientSet.PrependWatchReactor("pods", core.DefaultWatchReactor(kube1PodWatch, nil))
	kube2ClientSet := kubeclientfake.NewSimpleClientset()
	kube2RsWatch := watch.NewFake()
	kube2ClientSet.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube2RsWatch, nil))
	kube2PodWatch := watch.NewFake()
	kube2ClientSet.PrependWatchReactor("pods", core.DefaultWatchReactor(kube2PodWatch, nil))

	fedInformerClientFactory := func(cluster *fedv1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case "k8s-1":
			return kube1ClientSet, nil
		case "k8s-2":
			return kube2ClientSet, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}
	replicaSetController := NewReplicaSetController(fedClientSet)
	rsFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedReplicaSetInformer)
	rsFedinformer.SetClientFactory(fedInformerClientFactory)
	podFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedPodInformer)
	podFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	go replicaSetController.Run(1, stopChan)

	rs := newReplicaSetWithReplicas("rs", 9)
	rs.Annotations = map[string]string{FedReplicaSetPuppetAnnotation: "true"}
	rs, _ = fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Create(rs)
	fedRsWatch.Add(rs)

	rs1 := newReplicaSetWithReplicas("rs", 5)
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1.Status.FullyLabeledReplicas = *rs1.Spec.Replicas
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs1.Status.AvailableReplicas = *rs1.Spec.Replicas
	rs1, _ = kube1ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Create(rs1)
	kube1RsWatch.Add(rs1)

	rs2 := newReplicaSetWithReplicas("rs", 2)
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2.Status.FullyLabeledReplicas = *rs2.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas
	rs2.Status.AvailableReplicas = *rs2.Spec.Replicas
	rs2, _ = kube2ClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Create(rs2)
	kube2RsWatch.Add(rs2)

	start := time.Now()
	ok := false
	lastProblem := ""

	isOk := func() (bool, string) {
		rs, _ := fedClientSet.Extensions().ReplicaSets(metav1.NamespaceDefault).Get(rs.Name, metav1.GetOptions{})

		if *rs1.Spec.Replicas+*rs2.Spec.Replicas != *rs.Spec.Replicas {
			return false, "Spec.Replicas"
		}
		if rs1.Status.Replicas+rs2.Status.Replicas != rs.Status.Replicas {
			return false, "Status.Replicas"
		}
		if rs1.Status.FullyLabeledReplicas+rs2.Status.FullyLabeledReplicas != rs.Status.FullyLabeledReplicas {
			return false, "Status.FullyLabeledReplicas"
		}
		if rs1.Status.ReadyReplicas+rs2.Status.ReadyReplicas != rs.Status.ReadyReplicas {
			return false, "Status.ReadyReplicas"
		}
		if rs1.Status.AvailableReplicas+rs2.Status.AvailableReplicas != rs.Status.AvailableReplicas {
			return false, "Status.AvailableReplicas"
		}
		return true, ""
	}

	for ; !ok && time.Now().Sub(start) < 5*time.Second; time.Sleep(100 * time.Millisecond) {
		ok, lastProblem = isOk()
	}
	assert.True(t, ok, lastProblem+" is wrong")
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
