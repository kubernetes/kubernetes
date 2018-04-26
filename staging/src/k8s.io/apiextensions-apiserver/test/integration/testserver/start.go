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

package testserver

import (
	"fmt"
	"net"
	"os"
	"time"

	"github.com/pborman/uuid"

	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/cmd/server"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

func DefaultServerConfig() (*extensionsapiserver.Config, error) {
	listener, port, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}

	options := server.NewCustomResourceDefinitionsServerOptions(os.Stdout, os.Stderr)
	options.RecommendedOptions.Audit.LogOptions.Path = "-"
	options.RecommendedOptions.SecureServing.BindPort = port
	options.RecommendedOptions.Authentication = nil // disable
	options.RecommendedOptions.Authorization = nil  // disable
	options.RecommendedOptions.Admission = nil      // disable
	options.RecommendedOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
	options.RecommendedOptions.SecureServing.Listener = listener
	etcdURL, ok := os.LookupEnv("KUBE_INTEGRATION_ETCD_URL")
	if !ok {
		etcdURL = "http://127.0.0.1:2379"
	}
	options.RecommendedOptions.Etcd.StorageConfig.ServerList = []string{etcdURL}
	options.RecommendedOptions.Etcd.StorageConfig.Prefix = uuid.New()

	genericConfig := genericapiserver.NewRecommendedConfig(extensionsapiserver.Codecs)

	if err := options.RecommendedOptions.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{net.ParseIP("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}
	if err := options.RecommendedOptions.ApplyTo(genericConfig, nil); err != nil {
		return nil, err
	}
	if err := options.APIEnablement.ApplyTo(&genericConfig.Config, extensionsapiserver.DefaultAPIResourceConfigSource(), extensionsapiserver.Registry); err != nil {
		return nil, err
	}

	customResourceDefinitionRESTOptionsGetter := extensionsapiserver.CRDRESTOptionsGetter{
		StorageConfig:           options.RecommendedOptions.Etcd.StorageConfig,
		StoragePrefix:           options.RecommendedOptions.Etcd.StorageConfig.Prefix,
		EnableWatchCache:        options.RecommendedOptions.Etcd.EnableWatchCache,
		DefaultWatchCacheSize:   options.RecommendedOptions.Etcd.DefaultWatchCacheSize,
		EnableGarbageCollection: options.RecommendedOptions.Etcd.EnableGarbageCollection,
		DeleteCollectionWorkers: options.RecommendedOptions.Etcd.DeleteCollectionWorkers,
	}
	customResourceDefinitionRESTOptionsGetter.StorageConfig.Codec = unstructured.UnstructuredJSONScheme

	config := &extensionsapiserver.Config{
		GenericConfig: genericConfig,
		ExtraConfig: extensionsapiserver.ExtraConfig{
			CRDRESTOptionsGetter: customResourceDefinitionRESTOptionsGetter,
		},
	}

	return config, nil
}

func StartServer(config *extensionsapiserver.Config) (chan struct{}, *rest.Config, error) {
	stopCh := make(chan struct{})
	server, err := config.Complete().New(genericapiserver.EmptyDelegate)
	if err != nil {
		return nil, nil, err
	}
	go func() {
		err := server.GenericAPIServer.PrepareRun().Run(stopCh)
		if err != nil {
			close(stopCh)
			panic(err)
		}
	}()

	// wait until the server is healthy
	err = wait.PollImmediate(30*time.Millisecond, 30*time.Second, func() (bool, error) {
		healthClient, err := clientset.NewForConfig(server.GenericAPIServer.LoopbackClientConfig)
		if err != nil {
			return false, nil
		}
		healthResult := healthClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do()
		if healthResult.Error() != nil {
			return false, nil
		}
		rawHealth, err := healthResult.Raw()
		if err != nil {
			return false, nil
		}
		if string(rawHealth) != "ok" {
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		close(stopCh)
		return nil, nil, err
	}

	return stopCh, config.GenericConfig.LoopbackClientConfig, nil
}

func StartDefaultServer() (chan struct{}, *rest.Config, error) {
	config, err := DefaultServerConfig()
	if err != nil {
		return nil, nil, err
	}

	return StartServer(config)
}

func StartDefaultServerWithClients() (chan struct{}, clientset.Interface, dynamic.ClientPool, error) {
	stopCh, config, err := StartDefaultServer()
	if err != nil {
		return nil, nil, nil, err
	}

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		close(stopCh)
		return nil, nil, nil, err
	}

	return stopCh, apiExtensionsClient, dynamic.NewDynamicClientPool(config), nil
}
