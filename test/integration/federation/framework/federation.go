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

package framework

import (
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	clustercontroller "k8s.io/kubernetes/federation/pkg/federation-controller/cluster"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/e2e_node/services"
	"k8s.io/kubernetes/test/integration/framework"
)

type MemberCluster struct {
	CloseFn             framework.CloseFunc
	Config              *master.Config
	Client              clientset.Interface
	Host                string
	namespaceController *services.NamespaceController
}

// FederationFixture manages a federation api server and a set of member clusters
type FederationFixture struct {
	APIFixture          *FederationAPIFixture
	DesiredClusterCount int
	Clusters            []*MemberCluster

	ClusterClients    []clientset.Interface
	ClusterController *clustercontroller.ClusterController
	fedClient         federationclientset.Interface
	stopChan          chan struct{}
}

func (f *FederationFixture) SetUp(t *testing.T) {
	if f.APIFixture != nil {
		t.Fatal("Fixture already started")
	}
	if f.DesiredClusterCount < 1 {
		f.DesiredClusterCount = 1
	}
	defer TearDownOnPanic(t, f)

	t.Logf("Starting a federation of %d clusters", f.DesiredClusterCount)

	f.APIFixture = &FederationAPIFixture{}
	runOptions := GetRunOptions()
	// Enable all apis features for test.
	runOptions.APIEnablement.RuntimeConfig.Set("api/all=true")
	f.APIFixture.SetUpWithRunOptions(t, runOptions)

	f.stopChan = make(chan struct{})
	monitorPeriod := 1 * time.Second
	clustercontroller.StartClusterController(f.APIFixture.NewConfig(), f.stopChan, monitorPeriod)

	f.fedClient = f.APIFixture.NewClient("federation-fixture")
	for i := 0; i < f.DesiredClusterCount; i++ {
		f.StartCluster(t)
	}
}

func (f *FederationFixture) StartCluster(t *testing.T) {
	config := framework.NewMasterConfig()
	_, _, closeFn := framework.RunAMaster(config)
	host := config.GenericConfig.LoopbackClientConfig.Host

	clusterClient := clientset.NewForConfigOrDie(config.GenericConfig.LoopbackClientConfig)
	f.ClusterClients = append(f.ClusterClients, clusterClient)
	memberCluster := &MemberCluster{
		CloseFn:             closeFn,
		Config:              config,
		Client:              clusterClient,
		Host:                host,
		namespaceController: services.NewNamespaceController(host),
	}
	f.Clusters = append(f.Clusters, memberCluster)
	err := memberCluster.namespaceController.Start()
	if err != nil {
		t.Fatal(err)
	}

	clusterId := len(f.ClusterClients)

	t.Logf("Federated cluster %d serving on %s", clusterId, host)

	cluster := &federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:   fmt.Sprintf("cluster-%d", clusterId),
			Labels: map[string]string{"cluster": fmt.Sprintf("%d", clusterId)},
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: host,
				},
			},
			// Use insecure access
			SecretRef: nil,
		},
	}
	f.fedClient.FederationV1beta1().Clusters().Create(cluster)
}

func (f *FederationFixture) TearDown(t *testing.T) {
	if f.stopChan != nil {
		close(f.stopChan)
		f.stopChan = nil
	}
	for _, cluster := range f.Clusters {
		// Need to close controllers with active connections to the
		// cluster api before stopping the api or the connections will
		// hang until tcp timeout.
		cluster.namespaceController.Stop()
		cluster.CloseFn()
	}
	f.Clusters = nil
	if f.APIFixture != nil {
		f.APIFixture.TearDown(t)
		f.APIFixture = nil
	}
}
