/*
Copyright 2014 The Kubernetes Authors.

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
package genericcontrolplane

import (
	"errors"
	"fmt"
	"net/url"
	"time"

	kcpkubernetesclientset "github.com/kcp-dev/client-go/kubernetes"
	kcpkubernetesinformers "github.com/kcp-dev/client-go/informers"
	"github.com/kcp-dev/logicalcluster/v2"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/clientsethack"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/informerfactoryhack"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/notfoundhandler"
	"k8s.io/apiserver/pkg/util/openapi"
	"k8s.io/client-go/util/keyutil"
	_ "k8s.io/component-base/metrics/prometheus/workqueue"
	"k8s.io/component-base/version"
	"k8s.io/kubernetes/pkg/api/genericcontrolplanescheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericcontrolplane/aggregator"
	"k8s.io/kubernetes/pkg/genericcontrolplane/apis"
	"k8s.io/kubernetes/pkg/genericcontrolplane/clientutils"
	"k8s.io/kubernetes/pkg/genericcontrolplane/options"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

var LocalAdminCluster = logicalcluster.New("system:admin")

type ServerChain struct {
	CustomResourceDefinitions *extensionsapiserver.CustomResourceDefinitions
	GenericControlPlane       *apis.GenericControlPlane
	MiniAggregator            *aggregator.MiniAggregatorServer
}

// CreateServerChain creates the apiservers connected via delegation.
func CreateServerChain(miniAggregatorConfig aggregator.CompletedMiniAggregatorConfig, apisConfig apis.CompletedConfig, apiExtensionConfig extensionsapiserver.CompletedConfig) (*ServerChain, error) {
	notFoundHandler := notfoundhandler.New(apisConfig.GenericConfig.Serializer, genericapifilters.NoMuxAndDiscoveryIncompleteKey)
	apiExtensionsServer, err := apiExtensionConfig.New(genericapiserver.NewEmptyDelegateWithCustomHandler(notFoundHandler))
	if err != nil {
		return nil, fmt.Errorf("create api extensions: %v", err)
	}

	kubeAPIServer, err := apisConfig.New(apiExtensionsServer.GenericAPIServer)
	if err != nil {
		return nil, err
	}

	miniAggregatorServer, err := miniAggregatorConfig.New(kubeAPIServer.GenericAPIServer, kubeAPIServer, apiExtensionsServer)
	if err != nil {
		return nil, err
	}

	return &ServerChain{
		CustomResourceDefinitions: apiExtensionsServer,
		GenericControlPlane:       kubeAPIServer,
		MiniAggregator:            miniAggregatorServer,
	}, nil
}

// CreateKubeAPIServerConfig creates all the resources for running the API server, but runs none of them
func CreateKubeAPIServerConfig(
	genericConfig *genericapiserver.Config,
	o options.CompletedServerRunOptions,
	versionedInformers kcpkubernetesinformers.SharedInformerFactory,
	additionalPluginInitializers []admission.PluginInitializer,
	storageFactory *serverstorage.DefaultStorageFactory,
) (
	*apis.Config,
	error,
) {
	o.Metrics.Apply()
	serviceaccount.RegisterMetrics()

	// Load the public keys.
	var pubKeys []interface{}
	for _, f := range o.Authentication.ServiceAccounts.KeyFiles {
		keys, err := keyutil.PublicKeysFromFile(f)
		if err != nil {
			return nil, fmt.Errorf("failed to parse key file %q: %v", f, err)
		}
		pubKeys = append(pubKeys, keys...)
	}

	config := &apis.Config{
		GenericConfig: genericConfig,
		ExtraConfig: apis.ExtraConfig{
			APIResourceConfigSource: storageFactory.APIResourceConfigSource,
			StorageFactory:          storageFactory,
			EventTTL:                o.EventTTL,
			EnableLogsSupport:       o.EnableLogsHandler,

			VersionedInformers: versionedInformers,

			IdentityLeaseDurationSeconds:      o.IdentityLeaseDurationSeconds,
			IdentityLeaseRenewIntervalSeconds: o.IdentityLeaseRenewIntervalSeconds,

			ServiceAccountIssuer:        o.ServiceAccountIssuer,
			ServiceAccountMaxExpiration: o.ServiceAccountTokenMaxExpiration,
			ExtendExpiration:            o.Authentication.ServiceAccounts.ExtendExpiration,
			ServiceAccountIssuerURL:     o.Authentication.ServiceAccounts.Issuers[0], // panic?
			ServiceAccountJWKSURI:       o.Authentication.ServiceAccounts.JWKSURI,
			ServiceAccountPublicKeys:    pubKeys,
		},
	}

	clientCAProvider, err := o.Authentication.ClientCert.GetClientCAContentProvider()
	if err != nil {
		return nil, err
	}
	config.ExtraConfig.ClusterAuthenticationInfo.ClientCA = clientCAProvider

	requestHeaderConfig, err := o.Authentication.RequestHeader.ToAuthenticationRequestHeaderConfig()
	if err != nil {
		return nil, err
	}
	if requestHeaderConfig != nil {
		config.ExtraConfig.ClusterAuthenticationInfo.RequestHeaderCA = requestHeaderConfig.CAContentProvider
		config.ExtraConfig.ClusterAuthenticationInfo.RequestHeaderAllowedNames = requestHeaderConfig.AllowedClientNames
		config.ExtraConfig.ClusterAuthenticationInfo.RequestHeaderExtraHeaderPrefixes = requestHeaderConfig.ExtraHeaderPrefixes
		config.ExtraConfig.ClusterAuthenticationInfo.RequestHeaderGroupHeaders = requestHeaderConfig.GroupHeaders
		config.ExtraConfig.ClusterAuthenticationInfo.RequestHeaderUsernameHeaders = requestHeaderConfig.UsernameHeaders
	}

	client, err := kcpkubernetesclientset.NewForConfig(config.GenericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	if err := o.ServerRunOptions.Admission.ApplyTo(
		config.GenericConfig,
		informerfactoryhack.Wrap(config.ExtraConfig.VersionedInformers),
		clientsethack.Wrap(client),
		utilfeature.DefaultFeatureGate,
		additionalPluginInitializers...); err != nil {
		return nil, err
	}

	return config, nil
}

// BuildGenericConfig takes the master server options and produces the genericapiserver.Config associated with it
func BuildGenericConfig(
	o options.CompletedServerRunOptions,
) (
	genericConfig *genericapiserver.Config,
	storageFactory *serverstorage.DefaultStorageFactory,
	versionedInformers kcpkubernetesinformers.SharedInformerFactory,
	clientgoExternalClient kcpkubernetesclientset.ClusterInterface,
	lastErr error,
) {
	genericConfig = genericapiserver.NewConfig(genericcontrolplanescheme.Codecs)
	if lastErr = o.GenericServerRunOptions.ApplyTo(genericConfig); lastErr != nil {
		return
	}

	if lastErr = o.SecureServing.ApplyTo(&genericConfig.SecureServing, &genericConfig.LoopbackClientConfig); lastErr != nil {
		return
	}
	if lastErr = o.Features.ApplyTo(genericConfig); lastErr != nil {
		return
	}
	if lastErr = o.APIEnablement.ApplyTo(genericConfig, apis.DefaultAPIResourceConfigSource(), genericcontrolplanescheme.Scheme); lastErr != nil {
		return
	}
	if lastErr = o.EgressSelector.ApplyTo(genericConfig); lastErr != nil {
		return
	}
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.APIServerTracing) {
		if lastErr = o.Traces.ApplyTo(genericConfig.EgressSelector, genericConfig); lastErr != nil {
			return
		}
	}

	// wrap the definitions to revert any changes from disabled features
	getOpenAPIDefinitions := openapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(generatedopenapi.GetOpenAPIDefinitions)
	genericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(getOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme, extensionsapiserver.Scheme, extensionsapiserver.Scheme))
	genericConfig.OpenAPIConfig.Info.Title = "Kubernetes"
	genericConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion

	storageFactoryConfig := kubeapiserver.NewStorageFactoryConfig(genericcontrolplanescheme.Scheme, genericcontrolplanescheme.Codecs)
	storageFactoryConfig.APIResourceConfig = genericConfig.MergedResourceConfig
	completedStorageFactoryConfig, err := storageFactoryConfig.Complete(o.Etcd)
	if err != nil {
		lastErr = err
		return
	}
	storageFactory, lastErr = completedStorageFactoryConfig.New()
	if lastErr != nil {
		return
	}
	if genericConfig.EgressSelector != nil {
		storageFactory.StorageConfig.Transport.EgressLookup = genericConfig.EgressSelector.Lookup
	}
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.APIServerTracing) && genericConfig.TracerProvider != nil {
		storageFactory.StorageConfig.Transport.TracerProvider = genericConfig.TracerProvider
	}
	if lastErr = o.Etcd.ApplyWithStorageFactoryTo(storageFactory, genericConfig); lastErr != nil {
		return
	}

	// Use protobufs for self-communication.
	// Since not every generic apiserver has to support protobufs, we
	// cannot default to it in generic apiserver and need to explicitly
	// set it in kube-apiserver.
	genericConfig.LoopbackClientConfig.ContentConfig.ContentType = "application/vnd.kubernetes.protobuf"
	// Disable compression for self-communication, since we are going to be
	// on a fast local network
	genericConfig.LoopbackClientConfig.DisableCompression = true

	kubeClientConfig := genericConfig.LoopbackClientConfig
	clientutils.EnableMultiCluster(genericConfig.LoopbackClientConfig, genericConfig, "namespaces", "apiservices", "customresourcedefinitions", "clusterroles", "clusterrolebindings", "roles", "rolebindings", "serviceaccounts", "secrets")
	clientgoExternalClient, err = kcpkubernetesclientset.NewForConfig(kubeClientConfig)
	if err != nil {
		lastErr = fmt.Errorf("failed to create real external clientset: %v", err)
		return
	}
	versionedInformers = kcpkubernetesinformers.NewSharedInformerFactory(clientgoExternalClient, 10*time.Minute)

	// Authentication.ApplyTo requires already applied OpenAPIConfig and EgressSelector if present
	if lastErr = o.Authentication.ApplyTo(&genericConfig.Authentication, genericConfig.SecureServing, genericConfig.EgressSelector, genericConfig.OpenAPIConfig, genericConfig.OpenAPIV3Config, clientgoExternalClient, versionedInformers); lastErr != nil {
		return
	}

	return
}

// unimplementedServiceResolver is a webhook.ServiceResolver that always returns an error.
type unimplementedServiceResolver struct{}

// ResolveEndpoint always returns an error that this is not yet supported.
func (r *unimplementedServiceResolver) ResolveEndpoint(namespace string, name string, port int32) (*url.URL, error) {
	return nil, errors.New("webhook admission and  conversions are not yet supported in kcp")
}
