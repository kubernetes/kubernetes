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

package kubectl

import (
	"context"
	"encoding/json"
	"fmt"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	"net"
	"os"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest"
	clientcmdv1 "k8s.io/client-go/tools/clientcmd/api/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestKubectlFactoryShareTransport(t *testing.T) {
	// server & kubeconfig setup
	kubeConfigFile, err := os.CreateTemp(os.TempDir(), "kubectl_share_transport_test_temp")
	if err != nil {
		t.Fatalf(fmt.Sprintf("unable to create a client config: %v", err))
	}
	kubeConfig := kubeConfigFile.Name()
	defer os.Remove(kubeConfig)
	cacheDir, err := os.MkdirTemp(os.TempDir(), "kubectl_share_transport_test_cache_temp")
	if err != nil {
		t.Fatalf(fmt.Sprintf("unable to create a discovery cache dir: %v", err))
	}
	defer os.Remove(cacheDir)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	config := createKubeConfig(server)
	if err := json.NewEncoder(kubeConfigFile).Encode(config); err != nil {
		t.Fatal(err)
	}

	// test connections
	factory, testResult := newKubectlFactory(kubeConfig, cacheDir)

	// test multiple RESTClient invocations
	for i := 0; i < 2; i++ {
		restClient, err := factory.RESTClient()
		if err != nil {
			t.Fatal(err)
		}

		result := restClient.Get().AbsPath("/healthz").Do(context.TODO())
		_, err = result.Raw()
		if err != nil {
			t.Fatalf("unexpected error when obtaining /healthz: invocation %d: %v", i, err)
		}
	}

	// test ClientSet
	clientset, err := factory.KubernetesClientSet()
	if err != nil {
		t.Fatal(err)
	}

	_, err = clientset.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-share-transport"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create test ns: %v", err)
	}

	namespaces, err := clientset.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing namespaces: %v", err)
	}
	t.Logf("Listed %d namespaces on the cluster", len(namespaces.Items))

	// test another ClientSet
	clientset2, err := factory.KubernetesClientSet()
	if err != nil {
		t.Fatal(err)
	}

	pods, err := clientset2.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing pods: %v", err)
	}
	t.Logf("Listed %d pods on the cluster", len(pods.Items))

	// invoke another ClientSet manually
	clientConfig, err := factory.ToRESTConfig()
	if err != nil {
		t.Fatal(err)
	}
	httpClient, err := factory.ToHTTPClient()
	if err != nil {
		t.Fatal(err)
	}
	clientset3, err := kubernetes.NewForConfigAndClient(clientConfig, httpClient)
	if err != nil {
		t.Fatal(err)
	}

	rs, err := clientset3.AppsV1().ReplicaSets("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing replica sets: %v", err)
	}
	t.Logf("Listed %d replica sets on the cluster", len(rs.Items))

	// test multiple dynamic client invocations
	for i := 0; i < 2; i++ {
		dynamicClient, err := factory.DynamicClient()
		if err != nil {
			t.Fatal(err)
		}
		// check dynamic list
		resource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
		_, err = dynamicClient.Resource(resource).Namespace("default").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			t.Fatalf("unexpected error when listing pods with dynamic client: invocation %d: %v", i, err)
		}
	}

	if testResult.transportsCreated != 1 {
		t.Fatalf("expected only one transport, created %d transports", testResult.transportsCreated)
	}

	// using discovery client will bump the transportsCreated to 2 since it is hard to share the transport with CachedDiscoveryClient which needs to wrap the transport for caching

	discoveryClient, err := factory.ToDiscoveryClient()
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = discoveryClient.ServerGroupsAndResources()
	if err != nil {
		t.Fatal(err)
	}

	// test multiple RESTMapper (uses discoveryClient) and ClientForMapping (RESTClient) invocations
	for i := 0; i < 2; i++ {
		mapper, err := factory.ToRESTMapper() // uses discoveryClient
		if err != nil {
			t.Fatal(err)
		}
		mapping, err := mapper.RESTMapping(schema.GroupKind{Group: "apps", Kind: "ReplicaSet"}, "v1")
		if err != nil {
			t.Fatal(err)
		}

		restClient, err := factory.ClientForMapping(mapping)
		if err != nil {
			t.Fatal(err)
		}
		result := restClient.Get().Do(context.TODO())
		_, err = result.Raw()
		if err != nil {
			t.Fatalf("unexpected error when obtaining apps APIResourceList with RESTClient: invocation %d: %v", i, err)
		}
		unstructuredRestClient, err := factory.UnstructuredClientForMapping(mapping)
		if err != nil {
			t.Fatal(err)
		}
		result = unstructuredRestClient.Get().Do(context.TODO())
		_, err = result.Raw()
		if err != nil {
			t.Fatalf("unexpected error when obtaining apps APIResourceList with unstructured RestClient: invocation %d: %v", i, err)
		}
	}

	// test Builder
	builderResult := factory.NewBuilder().
		Unstructured().
		NamespaceParam("test-share-transport").
		ResourceTypeOrNameArgs(true, "namespace/test-share-transport", "pods/test1", "secret/test2", "configmaps/test3").
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	_, _ = builderResult.Infos()

	// test retrieval of OpenApiSchema
	_, err = factory.OpenAPIGetter().OpenAPISchema()
	if err != nil {
		t.Fatal(err)
	}
	_, err = factory.OpenAPISchema()
	if err != nil {
		t.Fatal(err)
	}

	// test retrieval of Validator
	dynamicClient, err := factory.DynamicClient()
	if err != nil {
		t.Fatal(err)
	}
	fieldValidationVerifier := resource.NewQueryParamVerifier(dynamicClient, factory.OpenAPIGetter(), resource.QueryParamFieldValidation)
	_, err = factory.Validator(metav1.FieldValidationStrict, fieldValidationVerifier)
	if err != nil {
		t.Fatal(err)
	}

	if testResult.transportsCreated != 2 {
		t.Fatalf("expected only 2 transports, created %d transports", testResult.transportsCreated)
	}

	// stdin should reset the clients and number of transport should increase
	factory.NewBuilder().StdinInUse()

	clientset4, err := factory.KubernetesClientSet()
	if err != nil {
		t.Fatal(err)
	}
	_, err = clientset4.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error when listing namespaces: %v", err)
	}

	if testResult.transportsCreated != 3 {
		t.Fatalf("expected only 3 transports, created %d transports", testResult.transportsCreated)
	}
}

type AfterResult struct {
	transportsCreated int
}

func newKubectlFactory(kubeconfigPath, cachePath string) (cmdutil.Factory, *AfterResult) {
	var result AfterResult
	var mu sync.Mutex

	dialFn := func(ctx context.Context, network, address string) (net.Conn, error) {
		mu.Lock()
		result.transportsCreated++
		mu.Unlock()
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}

	kubeConfigFlags := genericclioptions.NewConfigFlags(true).
		WithDeprecatedPasswordFlag().
		WithDiscoveryBurst(300).
		WithDiscoveryQPS(50.0).
		WithWrapConfigFn(func(config *rest.Config) *rest.Config {
			config.Dial = dialFn
			return config
		})
	kubeConfigFlags.KubeConfig = &kubeconfigPath
	kubeConfigFlags.CacheDir = &cachePath

	f := cmdutil.NewFactory(kubeConfigFlags)

	return f, &result

}

func createKubeConfig(server *kubeapiservertesting.TestServer) clientcmdv1.Config {
	clusterNick := "cluster"
	userNick := "user"
	contextNick := "context"
	return clientcmdv1.Config{
		Clusters: []clientcmdv1.NamedCluster{
			{
				Name: clusterNick,
				Cluster: clientcmdv1.Cluster{
					Server:                server.ClientConfig.Host,
					InsecureSkipTLSVerify: true,
				},
			},
		},
		AuthInfos: []clientcmdv1.NamedAuthInfo{
			{
				Name: userNick,
				AuthInfo: clientcmdv1.AuthInfo{
					Token: server.ClientConfig.BearerToken,
				},
			},
		},
		Contexts: []clientcmdv1.NamedContext{
			{
				Name: contextNick,
				Context: clientcmdv1.Context{
					Cluster:  clusterNick,
					AuthInfo: userNick,
				},
			},
		},
		CurrentContext: contextNick,
	}
}
