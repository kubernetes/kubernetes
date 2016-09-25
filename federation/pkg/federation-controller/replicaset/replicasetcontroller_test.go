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

	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/meta"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/watch"

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
	flag.Set("logtostderr", "true")
	flag.Set("v", "5")
	flag.Parse()

	replicaSetReviewDelay = 10 * time.Millisecond
	clusterAvailableDelay = 20 * time.Millisecond
	clusterUnavailableDelay = 60 * time.Millisecond
	allReplicaSetReviewDelay = 120 * time.Millisecond

	fedclientset := fedclientfake.NewSimpleClientset()
	fedrswatch := watch.NewFake()
	fedclientset.PrependWatchReactor("replicasets", core.DefaultWatchReactor(fedrswatch, nil))

	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-1", apiv1.ConditionTrue))
	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-2", apiv1.ConditionTrue))

	kube1clientset := kubeclientfake.NewSimpleClientset()
	kube1rswatch := watch.NewFake()
	kube1clientset.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube1rswatch, nil))
	kube1Podwatch := watch.NewFake()
	kube1clientset.PrependWatchReactor("pods", core.DefaultWatchReactor(kube1Podwatch, nil))
	kube2clientset := kubeclientfake.NewSimpleClientset()
	kube2rswatch := watch.NewFake()
	kube2clientset.PrependWatchReactor("replicasets", core.DefaultWatchReactor(kube2rswatch, nil))
	kube2Podwatch := watch.NewFake()
	kube2clientset.PrependWatchReactor("pods", core.DefaultWatchReactor(kube2Podwatch, nil))

	fedInformerClientFactory := func(cluster *fedv1.Cluster) (kube_release_1_4.Interface, error) {
		switch cluster.Name {
		case "k8s-1":
			return kube1clientset, nil
		case "k8s-2":
			return kube2clientset, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}
	replicaSetController := NewReplicaSetController(fedclientset)
	rsFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedReplicaSetInformer)
	rsFedinformer.SetClientFactory(fedInformerClientFactory)
	podFedinformer := testutil.ToFederatedInformerForTestOnly(replicaSetController.fedPodInformer)
	podFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	go replicaSetController.Run(1, stopChan)

	rs := newReplicaSetWithReplicas("rs", 9)
	rs, _ = fedclientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Create(rs)
	fedrswatch.Add(rs)
	time.Sleep(1 * time.Second)

	rs1, _ := kube1clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	kube1rswatch.Add(rs1)
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1, _ = kube1clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).UpdateStatus(rs1)
	kube1rswatch.Modify(rs1)

	rs2, _ := kube2clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	kube2rswatch.Add(rs2)
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2, _ = kube2clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).UpdateStatus(rs2)
	kube2rswatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedclientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)

	var replicas int32 = 20
	rs.Spec.Replicas = &replicas
	rs, _ = fedclientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Update(rs)
	fedrswatch.Modify(rs)
	time.Sleep(1 * time.Second)

	rs1, _ = kube1clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	rs1.Status.Replicas = *rs1.Spec.Replicas
	rs1, _ = kube1clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).UpdateStatus(rs1)
	kube1rswatch.Modify(rs1)

	rs2, _ = kube2clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	rs2.Status.Replicas = *rs2.Spec.Replicas
	rs2, _ = kube2clientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).UpdateStatus(rs2)
	kube2rswatch.Modify(rs2)

	time.Sleep(1 * time.Second)
	rs, _ = fedclientset.Extensions().ReplicaSets(apiv1.NamespaceDefault).Get(rs.Name)
	assert.Equal(t, *rs.Spec.Replicas, *rs1.Spec.Replicas+*rs2.Spec.Replicas)
	assert.Equal(t, rs.Status.Replicas, rs1.Status.Replicas+rs2.Status.Replicas)
}

func newReplicaSetWithReplicas(name string, replicas int32) *extensionsv1.ReplicaSet {
	return &extensionsv1.ReplicaSet{
		ObjectMeta: apiv1.ObjectMeta{
			Name:      name,
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/replicasets/name",
		},
		Spec: extensionsv1.ReplicaSetSpec{
			Replicas: &replicas,
		},
	}
}
