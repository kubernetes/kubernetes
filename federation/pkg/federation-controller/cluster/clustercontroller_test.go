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

package cluster

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	federationv1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	controllerutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
)

func newCluster(clusterName string, serverUrl string) *federationv1beta1.Cluster {
	cluster := federationv1beta1.Cluster{
		TypeMeta: metav1.TypeMeta{APIVersion: testapi.Federation.GroupVersion().String()},
		ObjectMeta: metav1.ObjectMeta{
			UID:  uuid.NewUUID(),
			Name: clusterName,
		},
		Spec: federationv1beta1.ClusterSpec{
			ServerAddressByClientCIDRs: []federationv1beta1.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: serverUrl,
				},
			},
		},
	}
	return &cluster
}

func newClusterList(cluster *federationv1beta1.Cluster) *federationv1beta1.ClusterList {
	clusterList := federationv1beta1.ClusterList{
		TypeMeta: metav1.TypeMeta{APIVersion: testapi.Federation.GroupVersion().String()},
		ListMeta: metav1.ListMeta{
			SelfLink: "foobar",
		},
		Items: []federationv1beta1.Cluster{},
	}
	clusterList.Items = append(clusterList.Items, *cluster)
	return &clusterList
}

// init a fake http handler, simulate a federation apiserver, response the "DELETE" "PUT" "GET" "UPDATE"
// when "canBeGotten" is false, means that user can not get the cluster cluster from apiserver
func createHttptestFakeHandlerForFederation(clusterList *federationv1beta1.ClusterList, canBeGotten bool) *http.HandlerFunc {
	fakeHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		clusterListString, _ := json.Marshal(*clusterList)
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case "PUT":
			fmt.Fprintln(w, string(clusterListString))
		case "GET":
			if canBeGotten {
				fmt.Fprintln(w, string(clusterListString))
			} else {
				fmt.Fprintln(w, "")
			}
		default:
			fmt.Fprintln(w, "")
		}
	})
	return &fakeHandler
}

// init a fake http handler, simulate a cluster apiserver, response the "/healthz"
// when "canBeGotten" is false, means that user can not get response from apiserver
func createHttptestFakeHandlerForCluster(canBeGotten bool) *http.HandlerFunc {
	fakeHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case "GET":
			if canBeGotten {
				fmt.Fprintln(w, "ok")
			} else {
				w.WriteHeader(http.StatusNotFound)
			}
		default:
			fmt.Fprintln(w, "")
		}
	})
	return &fakeHandler
}

func TestUpdateClusterStatusOK(t *testing.T) {
	clusterName := "foobarCluster"
	// create dummy httpserver
	testClusterServer := httptest.NewServer(createHttptestFakeHandlerForCluster(true))
	defer testClusterServer.Close()
	federationCluster := newCluster(clusterName, testClusterServer.URL)
	federationClusterList := newClusterList(federationCluster)

	testFederationServer := httptest.NewServer(createHttptestFakeHandlerForFederation(federationClusterList, true))
	defer testFederationServer.Close()

	restClientCfg, err := clientcmd.BuildConfigFromFlags(testFederationServer.URL, "")
	if err != nil {
		t.Errorf("Failed to build client config")
	}
	federationClientSet := federationclientset.NewForConfigOrDie(restclient.AddUserAgent(restClientCfg, "cluster-controller"))

	// Override KubeconfigGetterForSecret to avoid having to setup service accounts and mount files with secret tokens.
	originalGetter := controllerutil.KubeconfigGetterForSecret
	controllerutil.KubeconfigGetterForSecret = func(s *api.Secret) clientcmd.KubeconfigGetter {
		return func() (*clientcmdapi.Config, error) {
			return &clientcmdapi.Config{}, nil
		}
	}

	manager := newClusterController(federationClientSet, 5)
	err = manager.UpdateClusterStatus()
	if err != nil {
		t.Errorf("Failed to Update Cluster Status: %v", err)
	}
	clusterStatus, found := manager.clusterClusterStatusMap[clusterName]
	if !found {
		t.Errorf("Failed to Update Cluster Status")
	} else {
		if (clusterStatus.Conditions[1].Status != v1.ConditionFalse) || (clusterStatus.Conditions[1].Type != federationv1beta1.ClusterOffline) {
			t.Errorf("Failed to Update Cluster Status")
		}
	}

	// Reset KubeconfigGetterForSecret
	controllerutil.KubeconfigGetterForSecret = originalGetter
}
