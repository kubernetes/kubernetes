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
// It can be configured and called directly or via the hyperkube cache.
package app

import (
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	authorizerunion "k8s.io/kubernetes/pkg/auth/authorizer/union"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	genericvalidation "k8s.io/kubernetes/pkg/genericapiserver/validation"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	authenticatorunion "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
)

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewServerRunOptions()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "federation-apiserver",
		Long: `The Kubernetes federation API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}
	return cmd
}

// Run runs the specified APIServer.  This should never exit.
func Run(s *options.ServerRunOptions) error {
	genericvalidation.VerifyEtcdServersList(s.GenericServerRunOptions)
	genericapiserver.DefaultAndValidateRunOptions(s.GenericServerRunOptions)
	genericConfig := genericapiserver.NewConfig(). // create the new config
							ApplyOptions(s.GenericServerRunOptions). // apply the options selected
							Complete()                               // set default values based on the known values

	if err := genericConfig.MaybeGenerateServingCerts(); err != nil {
		glog.Fatalf("Failed to generate service certificate: %v", err)
	}

	// TODO: register cluster federation resources here.
	resourceConfig := genericapiserver.NewResourceConfig()

	if s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize == 0 {
		// When size of cache is not explicitly set, set it to 50000
		s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize = 50000
	}
	storageGroupsToEncodingVersion, err := s.GenericServerRunOptions.StorageGroupsToEncodingVersion()
	if err != nil {
		glog.Fatalf("error generating storage version map: %s", err)
	}
	storageFactory, err := genericapiserver.BuildDefaultStorageFactory(
		s.GenericServerRunOptions.StorageConfig, s.GenericServerRunOptions.DefaultStorageMediaType, api.Codecs,
		genericapiserver.NewDefaultResourceEncodingConfig(), storageGroupsToEncodingVersion,
		[]unversioned.GroupVersionResource{}, resourceConfig, s.GenericServerRunOptions.RuntimeConfig)
	if err != nil {
		glog.Fatalf("error in initializing storage factory: %s", err)
	}

	for _, override := range s.GenericServerRunOptions.EtcdServersOverrides {
		tokens := strings.Split(override, "#")
		if len(tokens) != 2 {
			glog.Errorf("invalid value of etcd server overrides: %s", override)
			continue
		}

		apiresource := strings.Split(tokens[0], "/")
		if len(apiresource) != 2 {
			glog.Errorf("invalid resource definition: %s", tokens[0])
			continue
		}
		group := apiresource[0]
		resource := apiresource[1]
		groupResource := unversioned.GroupResource{Group: group, Resource: resource}

		servers := strings.Split(tokens[1], ";")
		storageFactory.SetEtcdLocation(groupResource, servers)
	}

	apiAuthenticator, securityDefinitions, err := authenticator.New(authenticator.AuthenticatorConfig{
		Anonymous:           s.GenericServerRunOptions.AnonymousAuth,
		AnyToken:            s.GenericServerRunOptions.EnableAnyToken,
		BasicAuthFile:       s.GenericServerRunOptions.BasicAuthFile,
		ClientCAFile:        s.GenericServerRunOptions.ClientCAFile,
		TokenAuthFile:       s.GenericServerRunOptions.TokenAuthFile,
		OIDCIssuerURL:       s.GenericServerRunOptions.OIDCIssuerURL,
		OIDCClientID:        s.GenericServerRunOptions.OIDCClientID,
		OIDCCAFile:          s.GenericServerRunOptions.OIDCCAFile,
		OIDCUsernameClaim:   s.GenericServerRunOptions.OIDCUsernameClaim,
		OIDCGroupsClaim:     s.GenericServerRunOptions.OIDCGroupsClaim,
		KeystoneURL:         s.GenericServerRunOptions.KeystoneURL,
		RequestHeaderConfig: s.GenericServerRunOptions.AuthenticationRequestHeaderConfig(),
	})
	if err != nil {
		glog.Fatalf("Invalid Authentication Config: %v", err)
	}

	privilegedLoopbackToken := uuid.NewRandom().String()
	selfClientConfig, err := s.GenericServerRunOptions.NewSelfClientConfig(privilegedLoopbackToken)
	if err != nil {
		glog.Fatalf("Failed to create clientset: %v", err)
	}
	client, err := s.GenericServerRunOptions.NewSelfClient(privilegedLoopbackToken)
	if err != nil {
		glog.Errorf("Failed to create clientset: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(client, 10*time.Minute)

	authorizationConfig := authorizer.AuthorizationConfig{
		PolicyFile:                  s.GenericServerRunOptions.AuthorizationPolicyFile,
		WebhookConfigFile:           s.GenericServerRunOptions.AuthorizationWebhookConfigFile,
		WebhookCacheAuthorizedTTL:   s.GenericServerRunOptions.AuthorizationWebhookCacheAuthorizedTTL,
		WebhookCacheUnauthorizedTTL: s.GenericServerRunOptions.AuthorizationWebhookCacheUnauthorizedTTL,
		RBACSuperUser:               s.GenericServerRunOptions.AuthorizationRBACSuperUser,
		InformerFactory:             sharedInformers,
	}
	authorizationModeNames := strings.Split(s.GenericServerRunOptions.AuthorizationMode, ",")
	apiAuthorizer, err := authorizer.NewAuthorizerFromAuthorizationConfig(authorizationModeNames, authorizationConfig)
	if err != nil {
		glog.Fatalf("Invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.GenericServerRunOptions.AdmissionControl, ",")

	// TODO(dims): We probably need to add an option "EnableLoopbackToken"
	if apiAuthenticator != nil {
		var uid = uuid.NewRandom().String()
		tokens := make(map[string]*user.DefaultInfo)
		tokens[privilegedLoopbackToken] = &user.DefaultInfo{
			Name:   user.APIServerUser,
			UID:    uid,
			Groups: []string{user.SystemPrivilegedGroup},
		}

		tokenAuthenticator := authenticator.NewAuthenticatorFromTokens(tokens)
		apiAuthenticator = authenticatorunion.New(tokenAuthenticator, apiAuthenticator)

		tokenAuthorizer := authorizer.NewPrivilegedGroups(user.SystemPrivilegedGroup)
		apiAuthorizer = authorizerunion.New(tokenAuthorizer, apiAuthorizer)
	}

	pluginInitializer := admission.NewPluginInitializer(sharedInformers, apiAuthorizer)

	admissionController, err := admission.NewFromPlugins(client, admissionControlPluginNames, s.GenericServerRunOptions.AdmissionControlConfigFile, pluginInitializer)
	if err != nil {
		glog.Fatalf("Failed to initialize plugins: %v", err)
	}

	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion
	genericConfig.LoopbackClientConfig = selfClientConfig
	genericConfig.Authenticator = apiAuthenticator
	genericConfig.Authorizer = apiAuthorizer
	genericConfig.AdmissionControl = admissionController
	genericConfig.APIResourceConfigSource = storageFactory.APIResourceConfigSource
	genericConfig.OpenAPIConfig.Definitions = openapi.OpenAPIDefinitions
	genericConfig.EnableOpenAPISupport = true
	genericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions

	// TODO: Move this to generic api server (Need to move the command line flag).
	if s.GenericServerRunOptions.EnableWatchCache {
		cachesize.InitializeWatchCacheSizes(s.GenericServerRunOptions.TargetRAMMB)
		cachesize.SetWatchCacheSizes(s.GenericServerRunOptions.WatchCacheSizes)
	}

	m, err := genericConfig.New()
	if err != nil {
		return err
	}

	routes.UIRedirect{}.Install(m.HandlerContainer)
	routes.Logs{}.Install(m.HandlerContainer)

	// TODO: Refactor this code to share it with kube-apiserver rather than duplicating it here.
	restOptionsFactory := restOptionsFactory{
		storageFactory:          storageFactory,
		enableGarbageCollection: s.GenericServerRunOptions.EnableGarbageCollection,
		deleteCollectionWorkers: s.GenericServerRunOptions.DeleteCollectionWorkers,
	}
	if s.GenericServerRunOptions.EnableWatchCache {
		restOptionsFactory.storageDecorator = registry.StorageWithCacher
	} else {
		restOptionsFactory.storageDecorator = generic.UndecoratedStorage
	}

	installFederationAPIs(m, restOptionsFactory)
	installCoreAPIs(s, m, restOptionsFactory)
	installExtensionsAPIs(m, restOptionsFactory)

	sharedInformers.Start(wait.NeverStop)
	m.PrepareRun().Run(wait.NeverStop)
	return nil
}

type restOptionsFactory struct {
	storageFactory          genericapiserver.StorageFactory
	storageDecorator        generic.StorageDecorator
	deleteCollectionWorkers int
	enableGarbageCollection bool
}

func (f restOptionsFactory) NewFor(resource unversioned.GroupResource) generic.RESTOptions {
	config, err := f.storageFactory.NewConfig(resource)
	if err != nil {
		glog.Fatalf("Unable to find storage config for %v, due to %v", resource, err.Error())
	}
	return generic.RESTOptions{
		StorageConfig:           config,
		Decorator:               f.storageDecorator,
		DeleteCollectionWorkers: f.deleteCollectionWorkers,
		EnableGarbageCollection: f.enableGarbageCollection,
		ResourcePrefix:          f.storageFactory.ResourcePrefix(resource),
	}
}
