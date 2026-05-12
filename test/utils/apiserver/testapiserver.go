/*
Copyright 2022 The Kubernetes Authors.

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

package apiserver

import (
	"path"
	"testing"

	"github.com/google/uuid"

	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	etcdserver "k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/cert"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/utils/kubeconfig"
)

// TestAPIServer provides access to a running apiserver instance.
type TestAPIServer struct {
	// ClientSet is already initialized to access the apiserver as admin.
	ClientSet clientset.Interface

	// KubeConfigFile is the absolute path for a kube.config file that
	// grants admin access to the apiserver.
	KubeConfigFile string
}

// StartAPIServer runs etcd and apiserver in the background in the same
// process. All resources get released automatically when the test
// completes. If startup fails, the test gets aborted.
func StartAPITestServer(t *testing.T) TestAPIServer {
	cfg := etcdserver.NewTestConfig(t)
	etcdClient := etcdserver.RunEtcd(t, cfg)
	storageConfig := storagebackend.NewDefaultConfig(path.Join(uuid.New().String(), "registry"), nil)
	storageConfig.Transport.ServerList = etcdClient.Endpoints()

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{}, storageConfig)
	t.Cleanup(server.TearDownFn)

	clientSet := clientset.NewForConfigOrDie(server.ClientConfig)

	kubeConfigFile := writeKubeConfigForWardleServerToKASConnection(t, server.ClientConfig)

	return TestAPIServer{
		ClientSet:      clientSet,
		KubeConfigFile: kubeConfigFile,
	}
}

func writeKubeConfigForWardleServerToKASConnection(t *testing.T, kubeClientConfig *rest.Config) string {
	// write a kubeconfig out for starting other API servers with delegated auth.  remember, no in-cluster config
	// the loopback client config uses a loopback cert with different SNI.  We need to use the "real"
	// cert, so we'll hope we aren't hacked during a unit test and instead load it from the server we started.
	wardleToKASKubeClientConfig := rest.CopyConfig(kubeClientConfig)
	wardleToKASKubeClientConfig.ServerName = "" // reset SNI to use the "real" cert

	servingCerts, _, err := cert.GetServingCertificatesForURL(wardleToKASKubeClientConfig.Host, "")
	if err != nil {
		t.Fatal(err)
	}
	encodedServing, err := cert.EncodeCertificates(servingCerts...)
	if err != nil {
		t.Fatal(err)
	}
	wardleToKASKubeClientConfig.CAData = encodedServing

	for _, v := range servingCerts {
		t.Logf("Client: Server public key is %v\n", dynamiccertificates.GetHumanCertDetail(v))
	}
	certs, err := cert.ParseCertsPEM(wardleToKASKubeClientConfig.CAData)
	if err != nil {
		t.Fatal(err)
	}
	for _, curr := range certs {
		t.Logf("CA bundle %v\n", dynamiccertificates.GetHumanCertDetail(curr))
	}

	adminKubeConfig := kubeconfig.CreateKubeConfig(wardleToKASKubeClientConfig)
	tmpDir := t.TempDir()
	kubeConfigFile := path.Join(tmpDir, "kube.config")
	if err := clientcmd.WriteToFile(*adminKubeConfig, kubeConfigFile); err != nil {
		t.Fatal(err)
	}

	return kubeConfigFile
}
