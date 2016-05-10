/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/federation/cmd/federated-apiserver/app/options"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/cachesize"
)

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewAPIServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "federated-apiserver",
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
func Run(s *options.APIServer) error {
	genericapiserver.DefaultAndValidateRunOptions(s.ServerRunOptions)

	// TODO: register cluster federation resources here.
	resourceConfig := genericapiserver.NewResourceConfig()

	storageGroupsToEncodingVersion, err := s.StorageGroupsToEncodingVersion()
	if err != nil {
		glog.Fatalf("error generating storage version map: %s", err)
	}
	storageFactory, err := genericapiserver.BuildDefaultStorageFactory(
		s.StorageConfig, s.DefaultStorageMediaType, api.Codecs,
		genericapiserver.NewDefaultResourceEncodingConfig(), storageGroupsToEncodingVersion,
		resourceConfig, s.RuntimeConfig)
	if err != nil {
		glog.Fatalf("error in initializing storage factory: %s", err)
	}

	for _, override := range s.EtcdServersOverrides {
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

	authenticator, err := authenticator.New(authenticator.AuthenticatorConfig{
		BasicAuthFile:     s.BasicAuthFile,
		ClientCAFile:      s.ClientCAFile,
		TokenAuthFile:     s.TokenAuthFile,
		OIDCIssuerURL:     s.OIDCIssuerURL,
		OIDCClientID:      s.OIDCClientID,
		OIDCCAFile:        s.OIDCCAFile,
		OIDCUsernameClaim: s.OIDCUsernameClaim,
		OIDCGroupsClaim:   s.OIDCGroupsClaim,
		KeystoneURL:       s.KeystoneURL,
	})
	if err != nil {
		glog.Fatalf("Invalid Authentication Config: %v", err)
	}

	authorizationModeNames := strings.Split(s.AuthorizationMode, ",")
	authorizer, err := apiserver.NewAuthorizerFromAuthorizationConfig(authorizationModeNames, s.AuthorizationConfig)
	if err != nil {
		glog.Fatalf("Invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.AdmissionControl, ",")
	client, err := s.NewSelfClient()
	if err != nil {
		glog.Errorf("Failed to create clientset: %v", err)
	}
	admissionController := admission.NewFromPlugins(client, admissionControlPluginNames, s.AdmissionControlConfigFile)

	genericConfig := genericapiserver.NewConfig(s.ServerRunOptions)
	// TODO: Move the following to generic api server as well.
	genericConfig.StorageFactory = storageFactory
	genericConfig.Authenticator = authenticator
	genericConfig.SupportsBasicAuth = len(s.BasicAuthFile) > 0
	genericConfig.Authorizer = authorizer
	genericConfig.AdmissionControl = admissionController
	genericConfig.APIResourceConfigSource = storageFactory.APIResourceConfigSource
	genericConfig.MasterServiceNamespace = s.MasterServiceNamespace
	genericConfig.Serializer = api.Codecs

	// TODO: Move this to generic api server (Need to move the command line flag).
	if s.EnableWatchCache {
		cachesize.SetWatchCacheSizes(s.WatchCacheSizes)
	}

	m, err := genericapiserver.New(genericConfig)
	if err != nil {
		return err
	}

	installFederationAPIs(s, m, storageFactory)

	m.Run(s.ServerRunOptions)
	return nil
}
