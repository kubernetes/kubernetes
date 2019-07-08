/*
Copyright 2019 The Kubernetes Authors.

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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"time"

	"github.com/go-openapi/spec"

	utilwait "k8s.io/apimachinery/pkg/util/wait"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/preflight"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/capabilities"
	kubeapiserveropenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/master/tunneler"
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus" // for workqueue metric registration
)

const (
	etcdRetryLimit    = 60
	etcdRetryInterval = 1 * time.Second
)

// CreateKubeAPIServer creates and wires a workable kube-apiserver
func CreateKubeAPIServer(kubeAPIServerConfig *master.Config, delegateAPIServer genericapiserver.DelegationTarget, admissionPostStartHook genericapiserver.PostStartHookFunc) (*master.Master, error) {
	kubeAPIServer, err := kubeAPIServerConfig.Complete().New(delegateAPIServer)
	if err != nil {
		return nil, err
	}

	kubeAPIServer.GenericAPIServer.AddPostStartHookOrDie("start-kube-apiserver-admission-initializer", admissionPostStartHook)

	return kubeAPIServer, nil
}

// CreateKubeAPIServerConfig creates all the resources for running the API server, but runs none of them
func CreateKubeAPIServerConfig(
	s completedServerRunOptions,
	recommendedConfig genericapiserver.RecommendedConfig,
	storageFactory *serverstorage.DefaultStorageFactory,
	nodeTunneler tunneler.Tunneler,
	proxyTransport *http.Transport,
	securityDefinitions *spec.SecurityDefinitions,
) (
	c *master.Config,
	lastErr error,
) {
	if _, port, err := net.SplitHostPort(s.Etcd.StorageConfig.Transport.ServerList[0]); err == nil && port != "0" && len(port) != 0 {
		if err := utilwait.PollImmediate(etcdRetryInterval, etcdRetryLimit*etcdRetryInterval, preflight.EtcdConnection{ServerList: s.Etcd.StorageConfig.Transport.ServerList}.CheckEtcdServers); err != nil {
			lastErr = fmt.Errorf("error waiting for etcd connection: %v", err)
			return
		}
	}

	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: s.AllowPrivileged,
		// TODO(vmarmol): Implement support for HostNetworkSources.
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: []string{},
			HostPIDSources:     []string{},
			HostIPCSources:     []string{},
		},
		PerConnectionBandwidthLimitBytesPerSec: s.MaxConnectionBytesPerSec,
	})

	serviceIPRange, apiServerServiceIP, lastErr := master.DefaultServiceIPRange(s.ServiceClusterIPRange)
	if lastErr != nil {
		return
	}

	clientCA, lastErr := readCAorNil(s.Authentication.ClientCert.ClientCA)
	if lastErr != nil {
		return
	}
	requestHeaderProxyCA, lastErr := readCAorNil(s.Authentication.RequestHeader.ClientCAFile)
	if lastErr != nil {
		return
	}

	c = &master.Config{
		GenericConfig: &recommendedConfig.Config,
		ExtraConfig: master.ExtraConfig{
			ClientCARegistrationHook: master.ClientCARegistrationHook{
				ClientCA:                         clientCA,
				RequestHeaderUsernameHeaders:     s.Authentication.RequestHeader.UsernameHeaders,
				RequestHeaderGroupHeaders:        s.Authentication.RequestHeader.GroupHeaders,
				RequestHeaderExtraHeaderPrefixes: s.Authentication.RequestHeader.ExtraHeaderPrefixes,
				RequestHeaderCA:                  requestHeaderProxyCA,
				RequestHeaderAllowedNames:        s.Authentication.RequestHeader.AllowedNames,
			},

			APIResourceConfigSource: storageFactory.APIResourceConfigSource,
			StorageFactory:          storageFactory,
			EventTTL:                s.EventTTL,
			KubeletClientConfig:     s.KubeletConfig,
			EnableLogsSupport:       s.EnableLogsHandler,
			ProxyTransport:          proxyTransport,

			Tunneler: nodeTunneler,

			ServiceIPRange:       serviceIPRange,
			APIServerServiceIP:   apiServerServiceIP,
			APIServerServicePort: 443,

			ServiceNodePortRange:      s.ServiceNodePortRange,
			KubernetesServiceNodePort: s.KubernetesServiceNodePort,

			EndpointReconcilerType: reconcilers.Type(s.EndpointReconcilerType),
			MasterCount:            s.MasterCount,

			ServiceAccountIssuer:        s.ServiceAccountIssuer,
			ServiceAccountMaxExpiration: s.ServiceAccountTokenMaxExpiration,

			VersionedInformers: recommendedConfig.SharedInformerFactory,
		},
	}
	c.GenericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(kubeapiserveropenapi.GetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme))
	c.GenericConfig.OpenAPIConfig.Info.Title = "kube-apiserver"
	c.GenericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions

	if nodeTunneler != nil {
		// Use the nodeTunneler's dialer to connect to the kubelet
		c.ExtraConfig.KubeletClientConfig.Dial = nodeTunneler.Dial
	}
	if c.GenericConfig.EgressSelector != nil {
		// Use the config.GenericConfig.EgressSelector lookup to find the dialer to connect to the kubelet
		c.ExtraConfig.KubeletClientConfig.Lookup = c.GenericConfig.EgressSelector.Lookup
	}

	return
}

func readCAorNil(file string) ([]byte, error) {
	if len(file) == 0 {
		return nil, nil
	}
	return ioutil.ReadFile(file)
}
