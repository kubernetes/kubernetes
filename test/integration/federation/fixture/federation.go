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

package fixture

import (
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
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

type FederationFixture struct {
	APIFixture        *FederationAPIFixture
	ClusterCount      int
	Clusters          []*MemberCluster
	ClusterController *clustercontroller.ClusterController
}

func (f *FederationFixture) Setup(t *testing.T) {
	if f.ClusterCount < 1 {
		f.ClusterCount = 1
	}
	if f.APIFixture != nil {
		t.Fatal("Fixture already started")
	}
	f.APIFixture = &FederationAPIFixture{}
	f.APIFixture.Setup(t)

	// Start the cluster controller
	clusterClient := federationclientset.NewForConfigOrDie(f.APIFixture.NewRestConfig("cluster-controller"))
	f.ClusterController = clustercontroller.NewclusterController(clusterClient, 1*time.Second)
	// TODO stop at the end of the test
	f.ClusterController.Run()

	f.startClusters()
}

func (f *FederationFixture) startClusters() {
	for i := 0; i < f.ClusterCount; i++ {
		config := framework.NewMasterConfig()
		_, server := framework.RunAMaster(config)
		client := clientset.NewForConfigOrDie(config.GenericConfig.LoopbackClientConfig)
		host := config.GenericConfig.LoopbackClientConfig.Host
		f.Clusters = append(f.Clusters, &MemberCluster{
			Server: server,
			Config: config,
			Client: client,
			Host:   host,
		})
		fmt.Printf("Cluster %d API up: ", host)

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
		f.APIFixture.Client.FederationV1beta1().Clusters().Create(cluster)
	}
}

func (f *FederationFixture) Teardown(t *testing.T) {
	f.APIFixture.Teardown(t)
	for _, cluster := range f.Clusters {
		cluster.Server.Close()
	}
}

func (f *FederationFixture) NewRestConfig(userAgent string) *restclient.Config {
	return f.APIFixture.NewRestConfig(userAgent)
}
