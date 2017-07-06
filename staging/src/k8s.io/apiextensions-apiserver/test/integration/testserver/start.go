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
	"strconv"
	"time"

	"github.com/pborman/uuid"

	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/cmd/server"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/dynamic"
)

func DefaultServerConfig() (*extensionsapiserver.Config, error) {
	port, err := FindFreeLocalPort()
	if err != nil {
		return nil, err
	}

	options := server.NewCustomResourceDefinitionsServerOptions(os.Stdout, os.Stderr)
	options.RecommendedOptions.Audit.LogOptions.Path = "-"
	options.RecommendedOptions.SecureServing.BindPort = port
	options.RecommendedOptions.Authentication.SkipInClusterLookup = true
	options.RecommendedOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
	etcdURL, ok := os.LookupEnv("KUBE_INTEGRATION_ETCD_URL")
	if !ok {
		etcdURL = "http://127.0.0.1:2379"
	}
	options.RecommendedOptions.Etcd.StorageConfig.ServerList = []string{etcdURL}
	options.RecommendedOptions.Etcd.StorageConfig.Prefix = uuid.New()

	// TODO stop copying this
	// because there isn't currently a way to disable authentication or authorization from options
	// explode options.Config here
	genericConfig := genericapiserver.NewConfig(extensionsapiserver.Codecs)
	genericConfig.Authenticator = nil
	genericConfig.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()

	if err := options.RecommendedOptions.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{net.ParseIP("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}
	if err := options.RecommendedOptions.Etcd.ApplyTo(genericConfig); err != nil {
		return nil, err
	}
	if err := options.RecommendedOptions.SecureServing.ApplyTo(genericConfig); err != nil {
		return nil, err
	}
	if err := options.RecommendedOptions.Audit.ApplyTo(genericConfig); err != nil {
		return nil, err
	}
	if err := options.RecommendedOptions.Features.ApplyTo(genericConfig); err != nil {
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
	customResourceDefinitionRESTOptionsGetter.StorageConfig.Copier = extensionsapiserver.UnstructuredCopier{}

	config := &extensionsapiserver.Config{
		GenericConfig:        genericConfig,
		CRDRESTOptionsGetter: customResourceDefinitionRESTOptionsGetter,
	}

	return config, nil
}

func StartServer(config *extensionsapiserver.Config) (chan struct{}, clientset.Interface, dynamic.ClientPool, error) {
	stopCh := make(chan struct{})
	server, err := config.Complete().New(genericapiserver.EmptyDelegate)
	if err != nil {
		return nil, nil, nil, err
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
		return nil, nil, nil, err
	}

	apiExtensionsClient, err := clientset.NewForConfig(server.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		close(stopCh)
		return nil, nil, nil, err
	}

	bytes, _ := apiExtensionsClient.Discovery().RESTClient().Get().AbsPath("/apis/apiextensions.k8s.io/v1beta1").DoRaw()
	fmt.Print(string(bytes))

	return stopCh, apiExtensionsClient, dynamic.NewDynamicClientPool(server.GenericAPIServer.LoopbackClientConfig), nil
}

func StartDefaultServer() (chan struct{}, clientset.Interface, dynamic.ClientPool, error) {
	config, err := DefaultServerConfig()
	if err != nil {
		return nil, nil, nil, err
	}

	return StartServer(config)
}

// FindFreeLocalPort returns the number of an available port number on
// the loopback interface.  Useful for determining the port to launch
// a server on.  Error handling required - there is a non-zero chance
// that the returned port number will be bound by another process
// after this function returns.
func FindFreeLocalPort() (int, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	_, portStr, err := net.SplitHostPort(l.Addr().String())
	if err != nil {
		return 0, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return 0, err
	}
	return port, nil
}
