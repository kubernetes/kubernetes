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
	"net/http/httptest"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	clustercontroller "k8s.io/kubernetes/federation/pkg/federation-controller/cluster"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

type MemberCluster struct {
	Server *httptest.Server
	Config *master.Config
	Client clientset.Interface
	Host   string
}

// FederationFixture manages a federation api server and a set of member clusters
type FederationFixture struct {
	APIFixture          *FederationAPIFixture
	DesiredClusterCount int
	Clusters            []*MemberCluster
	ClusterClients      []clientset.Interface
	ClusterController   *clustercontroller.ClusterController
	stopChan            chan struct{}
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
	f.APIFixture.SetUp(t)

	f.stopChan = make(chan struct{})
	monitorPeriod := 1 * time.Second
	clustercontroller.StartClusterController(f.APIFixture.NewConfig(), f.stopChan, monitorPeriod)

	f.startClusters()
}

func (f *FederationFixture) startClusters() {
	fedClient := f.APIFixture.NewClient("federation-fixture")
	for i := 0; i < f.DesiredClusterCount; i++ {
		config := framework.NewMasterConfig()
		_, server := framework.RunAMaster(config)
		host := config.GenericConfig.LoopbackClientConfig.Host

		// Use fmt to ensure the output will be visible when run with go test -v
		fmt.Printf("Federated cluster %d serving on %s", i, host)

		clusterClient := clientset.NewForConfigOrDie(config.GenericConfig.LoopbackClientConfig)
		f.Clusters = append(f.Clusters, &MemberCluster{
			Server: server,
			Config: config,
			Client: clusterClient,
			Host:   host,
		})

		f.ClusterClients = append(f.ClusterClients, clusterClient)

		cluster := &federationapi.Cluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("cluster-%d", i),
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
		fedClient.FederationV1beta1().Clusters().Create(cluster)
	}
}

func (f *FederationFixture) TearDown(t *testing.T) {
	if f.stopChan != nil {
		close(f.stopChan)
		f.stopChan = nil
	}
	for _, cluster := range f.Clusters {
		cluster.Server.Close()
	}
	f.Clusters = nil
	if f.APIFixture != nil {
		f.APIFixture.TearDown(t)
		f.APIFixture = nil
	}
}
