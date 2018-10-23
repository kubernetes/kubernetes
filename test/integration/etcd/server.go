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

package etcd

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"

	// install all APIs
	_ "k8s.io/kubernetes/pkg/master"
)

// StartRealMasterOrDie starts an API master that is appropriate for use in tests that require one of every resource
func StartRealMasterOrDie(t *testing.T) *Master {
	certDir, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}

	_, defaultServiceClusterIPRange, err := net.ParseCIDR("10.0.0.0/24")
	if err != nil {
		t.Fatal(err)
	}

	listener, _, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerOptions := options.NewServerRunOptions()
	kubeAPIServerOptions.InsecureServing.BindPort = 0
	kubeAPIServerOptions.SecureServing.Listener = listener
	kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
	kubeAPIServerOptions.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURL()}
	kubeAPIServerOptions.Etcd.DefaultStorageMediaType = runtime.ContentTypeJSON // force json we can easily interpret the result in etcd
	kubeAPIServerOptions.ServiceClusterIPRange = *defaultServiceClusterIPRange
	kubeAPIServerOptions.Authorization.Modes = []string{"RBAC"}
	kubeAPIServerOptions.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
	completedOptions, err := app.Complete(kubeAPIServerOptions)
	if err != nil {
		t.Fatal(err)
	}
	if err := completedOptions.APIEnablement.RuntimeConfig.Set("api/all=true"); err != nil {
		t.Fatal(err)
	}

	// get etcd client before starting API server
	rawClient, kvClient, err := integration.GetEtcdClients(completedOptions.Etcd.StorageConfig)
	if err != nil {
		t.Fatal(err)
	}

	// get a leased session
	session, err := concurrency.NewSession(rawClient)
	if err != nil {
		t.Fatal(err)
	}

	// then build and use an etcd lock
	// this prevents more than one of these masters from running at the same time
	lock := concurrency.NewLocker(session, "kube_integration_etcd_raw")
	lock.Lock()

	// make sure we start with a clean slate
	if _, err := kvClient.Delete(context.Background(), "/registry/", clientv3.WithPrefix()); err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})

	kubeAPIServer, err := app.CreateServerChain(completedOptions, stopCh)
	if err != nil {
		t.Fatal(err)
	}

	kubeClientConfig := restclient.CopyConfig(kubeAPIServer.LoopbackClientConfig)

	// we make lots of requests, don't be slow
	kubeClientConfig.QPS = 99999
	kubeClientConfig.Burst = 9999

	kubeClient := clientset.NewForConfigOrDie(kubeClientConfig)

	go func() {
		// Catch panics that occur in this go routine so we get a comprehensible failure
		defer func() {
			if err := recover(); err != nil {
				t.Errorf("Unexpected panic trying to start API master: %#v", err)
			}
		}()

		if err := kubeAPIServer.PrepareRun().Run(stopCh); err != nil {
			t.Fatal(err)
		}
	}()

	lastHealth := ""
	if err := wait.PollImmediate(time.Second, time.Minute, func() (done bool, err error) {
		// wait for the server to be healthy
		result := kubeClient.RESTClient().Get().AbsPath("/healthz").Do()
		content, _ := result.Raw()
		lastHealth = string(content)
		if errResult := result.Error(); errResult != nil {
			t.Log(errResult)
			return false, nil
		}
		var status int
		result.StatusCode(&status)
		return status == http.StatusOK, nil
	}); err != nil {
		t.Log(lastHealth)
		t.Fatal(err)
	}

	// force cached discovery reset
	discoveryClient := cacheddiscovery.NewMemCacheClient(kubeClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()

	serverResources, err := kubeClient.Discovery().ServerResources()
	if err != nil {
		t.Fatal(err)
	}

	cleanup := func() {
		if err := os.RemoveAll(certDir); err != nil {
			t.Log(err)
		}
		close(stopCh)
		lock.Unlock()
	}

	return &Master{
		Client:    kubeClient,
		Dynamic:   dynamic.NewForConfigOrDie(kubeClientConfig),
		Config:    kubeClientConfig,
		KV:        kvClient,
		Mapper:    restMapper,
		Resources: GetResources(t, serverResources),
		Cleanup:   cleanup,
	}
}

// Master represents a running API server that is ready for use
// The Cleanup func must be deferred to prevent resource leaks
type Master struct {
	Client    clientset.Interface
	Dynamic   dynamic.Interface
	Config    *restclient.Config
	KV        clientv3.KV
	Mapper    meta.RESTMapper
	Resources []Resource
	Cleanup   func()
}

// Resource contains REST mapping information for a specific resource and extra metadata such as delete collection support
type Resource struct {
	Mapping             *meta.RESTMapping
	HasDeleteCollection bool
}

// GetResources fetches the Resources associated with serverResources that support get and create
func GetResources(t *testing.T, serverResources []*metav1.APIResourceList) []Resource {
	var resources []Resource

	for _, discoveryGroup := range serverResources {
		for _, discoveryResource := range discoveryGroup.APIResources {
			// this is a subresource, skip it
			if strings.Contains(discoveryResource.Name, "/") {
				continue
			}
			hasCreate := false
			hasGet := false
			hasDeleteCollection := false
			for _, verb := range discoveryResource.Verbs {
				if verb == "get" {
					hasGet = true
				}
				if verb == "create" {
					hasCreate = true
				}
				if verb == "deletecollection" {
					hasDeleteCollection = true
				}
			}
			if !(hasCreate && hasGet) {
				continue
			}

			resourceGV, err := schema.ParseGroupVersion(discoveryGroup.GroupVersion)
			if err != nil {
				t.Fatal(err)
			}
			gvk := resourceGV.WithKind(discoveryResource.Kind)
			if len(discoveryResource.Group) > 0 || len(discoveryResource.Version) > 0 {
				gvk = schema.GroupVersionKind{
					Group:   discoveryResource.Group,
					Version: discoveryResource.Version,
					Kind:    discoveryResource.Kind,
				}
			}
			gvr := resourceGV.WithResource(discoveryResource.Name)

			resources = append(resources, Resource{
				Mapping: &meta.RESTMapping{
					Resource:         gvr,
					GroupVersionKind: gvk,
					Scope:            scope(discoveryResource.Namespaced),
				},
				HasDeleteCollection: hasDeleteCollection,
			})
		}
	}

	return resources
}

func scope(namespaced bool) meta.RESTScope {
	if namespaced {
		return meta.RESTScopeNamespace
	}
	return meta.RESTScopeRoot
}

// JSONToUnstructured converts a JSON stub to unstructured.Unstructured and
// returns a dynamic resource client that can be used to interact with it
func JSONToUnstructured(stub, namespace string, mapping *meta.RESTMapping, dynamicClient dynamic.Interface) (dynamic.ResourceInterface, *unstructured.Unstructured, error) {
	typeMetaAdder := map[string]interface{}{}
	if err := json.Unmarshal([]byte(stub), &typeMetaAdder); err != nil {
		return nil, nil, err
	}

	// we don't require GVK on the data we provide, so we fill it in here.  We could, but that seems extraneous.
	typeMetaAdder["apiVersion"] = mapping.GroupVersionKind.GroupVersion().String()
	typeMetaAdder["kind"] = mapping.GroupVersionKind.Kind

	if mapping.Scope == meta.RESTScopeRoot {
		namespace = ""
	}

	return dynamicClient.Resource(mapping.Resource).Namespace(namespace), &unstructured.Unstructured{Object: typeMetaAdder}, nil
}
