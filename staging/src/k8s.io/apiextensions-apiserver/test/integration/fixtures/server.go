/*
Copyright 2018 The Kubernetes Authors.

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

package fixtures

import (
	"io/ioutil"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	serveroptions "k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	servertesting "k8s.io/apiextensions-apiserver/pkg/cmd/server/testing"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

// StartDefaultServer starts a test server.
func StartDefaultServer(t servertesting.Logger, flags ...string) (func(), *rest.Config, *options.CustomResourceDefinitionsServerOptions, error) {
	// create kubeconfig which will not actually be used. But authz/authn needs it to startup.
	fakeKubeConfig, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		return nil, nil, nil, err
	}
	fakeKubeConfig.WriteString(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: http://127.1.2.3:12345
  name: integration
contexts:
- context:
    cluster: integration
    user: test
  name: default-context
current-context: default-context
users:
- name: test
  user:
    password: test
    username: test
`)
	fakeKubeConfig.Close()

	s, err := servertesting.StartTestServer(t, nil, append([]string{
		"--etcd-prefix", uuid.New().String(),
		"--etcd-servers", strings.Join(IntegrationEtcdServers(), ","),
		"--authentication-skip-lookup",
		"--authentication-kubeconfig", fakeKubeConfig.Name(),
		"--authorization-kubeconfig", fakeKubeConfig.Name(),
		"--kubeconfig", fakeKubeConfig.Name(),
		"--disable-admission-plugins", "NamespaceLifecycle,MutatingAdmissionWebhook,ValidatingAdmissionWebhook"},
		flags...,
	), nil)
	if err != nil {
		os.Remove(fakeKubeConfig.Name())
		return nil, nil, nil, err
	}

	tearDownFn := func() {
		defer os.Remove(fakeKubeConfig.Name())
		s.TearDownFn()
	}

	return tearDownFn, s.ClientConfig, s.ServerOpts, nil
}

// StartDefaultServerWithClients starts a test server and returns clients for it.
func StartDefaultServerWithClients(t servertesting.Logger, extraFlags ...string) (func(), clientset.Interface, dynamic.Interface, error) {
	tearDown, config, _, err := StartDefaultServer(t, extraFlags...)
	if err != nil {
		return nil, nil, nil, err
	}

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		tearDown()
		return nil, nil, nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		tearDown()
		return nil, nil, nil, err
	}

	return tearDown, apiExtensionsClient, dynamicClient, nil
}

// StartDefaultServerWithClientsAndEtcd starts a test server and returns clients for it.
func StartDefaultServerWithClientsAndEtcd(t servertesting.Logger, extraFlags ...string) (func(), clientset.Interface, dynamic.Interface, *clientv3.Client, string, error) {
	tearDown, config, options, err := StartDefaultServer(t, extraFlags...)
	if err != nil {
		return nil, nil, nil, nil, "", err
	}

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		tearDown()
		return nil, nil, nil, nil, "", err
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		tearDown()
		return nil, nil, nil, nil, "", err
	}

	RESTOptionsGetter := serveroptions.NewCRDRESTOptionsGetter(*options.RecommendedOptions.Etcd)
	restOptions, err := RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: "hopefully-ignored-group", Resource: "hopefully-ignored-resources"})
	if err != nil {
		return nil, nil, nil, nil, "", err
	}
	tlsInfo := transport.TLSInfo{
		CertFile:      restOptions.StorageConfig.Transport.CertFile,
		KeyFile:       restOptions.StorageConfig.Transport.KeyFile,
		TrustedCAFile: restOptions.StorageConfig.Transport.TrustedCAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, nil, nil, nil, "", err
	}
	etcdConfig := clientv3.Config{
		Endpoints:   restOptions.StorageConfig.Transport.ServerList,
		DialTimeout: 20 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: tlsConfig,
	}
	etcdclient, err := clientv3.New(etcdConfig)
	if err != nil {
		return nil, nil, nil, nil, "", err
	}

	return tearDown, apiExtensionsClient, dynamicClient, etcdclient, restOptions.StorageConfig.Prefix, nil
}

// IntegrationEtcdServers returns etcd server URLs.
func IntegrationEtcdServers() []string {
	if etcdURL, ok := os.LookupEnv("KUBE_INTEGRATION_ETCD_URL"); ok {
		return []string{etcdURL}
	}
	return []string{"http://127.0.0.1:2379"}
}
