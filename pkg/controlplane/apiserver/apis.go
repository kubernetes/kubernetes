/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/client-go/discovery"
	"k8s.io/klog/v2"
	svmrest "k8s.io/kubernetes/pkg/registry/storagemigration/rest"

	admissionregistrationrest "k8s.io/kubernetes/pkg/registry/admissionregistration/rest"
	apiserverinternalrest "k8s.io/kubernetes/pkg/registry/apiserverinternal/rest"
	authenticationrest "k8s.io/kubernetes/pkg/registry/authentication/rest"
	authorizationrest "k8s.io/kubernetes/pkg/registry/authorization/rest"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	coordinationrest "k8s.io/kubernetes/pkg/registry/coordination/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	eventsrest "k8s.io/kubernetes/pkg/registry/events/rest"
	flowcontrolrest "k8s.io/kubernetes/pkg/registry/flowcontrol/rest"
	rbacrest "k8s.io/kubernetes/pkg/registry/rbac/rest"
)

// RESTStorageProvider is a factory type for REST storage.
type RESTStorageProvider interface {
	GroupName() string
	NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error)
}

// NewCoreGenericConfig returns a new core rest generic config.
func (c *CompletedConfig) NewCoreGenericConfig() *corerest.GenericConfig {
	return &corerest.GenericConfig{
		StorageFactory:              c.Extra.StorageFactory,
		EventTTL:                    c.Extra.EventTTL,
		LoopbackClientConfig:        c.Generic.LoopbackClientConfig,
		ServiceAccountIssuer:        c.Extra.ServiceAccountIssuer,
		ExtendExpiration:            c.Extra.ExtendExpiration,
		ServiceAccountMaxExpiration: c.Extra.ServiceAccountMaxExpiration,
		MaxExtendedExpiration:       c.Extra.ServiceAccountExtendedMaxExpiration,
		APIAudiences:                c.Generic.Authentication.APIAudiences,
		Informers:                   c.Extra.VersionedInformers,
	}
}

// GenericStorageProviders returns a set of APIs for a generic control plane.
// They ought to be a subset of those served by kube-apiserver.
func (c *CompletedConfig) GenericStorageProviders(discovery discovery.DiscoveryInterface) ([]RESTStorageProvider, error) {
	// The order here is preserved in discovery.
	// If resources with identical names exist in more than one of these groups (e.g. "deployments.apps"" and "deployments.extensions"),
	// the order of this list determines which group an unqualified resource name (e.g. "deployments") should prefer.
	// This priority order is used for local discovery, but it ends up aggregated in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go
	// with specific priorities.
	// TODO: describe the priority all the way down in the RESTStorageProviders and plumb it back through the various discovery
	// handlers that we have.
	return []RESTStorageProvider{
		c.NewCoreGenericConfig(),
		apiserverinternalrest.StorageProvider{},
		authenticationrest.RESTStorageProvider{Authenticator: c.Generic.Authentication.Authenticator, APIAudiences: c.Generic.Authentication.APIAudiences},
		authorizationrest.RESTStorageProvider{Authorizer: c.Generic.Authorization.Authorizer, RuleResolver: c.Generic.RuleResolver},
		certificatesrest.RESTStorageProvider{Authorizer: c.Generic.Authorization.Authorizer},
		coordinationrest.RESTStorageProvider{},
		rbacrest.RESTStorageProvider{Authorizer: c.Generic.Authorization.Authorizer},
		svmrest.RESTStorageProvider{},
		flowcontrolrest.RESTStorageProvider{InformerFactory: c.Generic.SharedInformerFactory},
		admissionregistrationrest.RESTStorageProvider{Authorizer: c.Generic.Authorization.Authorizer, DiscoveryClient: discovery},
		eventsrest.RESTStorageProvider{TTL: c.EventTTL},
	}, nil
}

// InstallAPIs will install the APIs for the restStorageProviders if they are enabled.
func (s *Server) InstallAPIs(restStorageProviders ...RESTStorageProvider) error {
	nonLegacy := []*genericapiserver.APIGroupInfo{}

	// used later in the loop to filter the served resource by those that have expired.
	resourceExpirationEvaluatorOpts := genericapiserver.ResourceExpirationEvaluatorOptions{
		CurrentVersion:                          s.GenericAPIServer.EffectiveVersion.EmulationVersion(),
		Prerelease:                              s.GenericAPIServer.EffectiveVersion.BinaryVersion().PreRelease(),
		EmulationForwardCompatible:              s.GenericAPIServer.EmulationForwardCompatible,
		RuntimeConfigEmulationForwardCompatible: s.GenericAPIServer.RuntimeConfigEmulationForwardCompatible,
	}
	resourceExpirationEvaluator, err := genericapiserver.NewResourceExpirationEvaluatorFromOptions(resourceExpirationEvaluatorOpts)
	if err != nil {
		return err
	}

	for _, restStorageBuilder := range restStorageProviders {
		groupName := restStorageBuilder.GroupName()
		apiGroupInfo, err := restStorageBuilder.NewRESTStorage(s.APIResourceConfigSource, s.RESTOptionsGetter)
		if err != nil {
			return fmt.Errorf("problem initializing API group %q: %w", groupName, err)
		}
		if len(apiGroupInfo.VersionedResourcesStorageMap) == 0 {
			// If we have no storage for any resource configured, this API group is effectively disabled.
			// This can happen when an entire API group, version, or development-stage (alpha, beta, GA) is disabled.
			klog.Infof("API group %q is not enabled, skipping.", groupName)
			continue
		}

		// Remove resources that serving kinds that are removed or not introduced yet at the current version.
		// We do this here so that we don't accidentally serve versions without resources or openapi information that for kinds we don't serve.
		// This is a spot above the construction of individual storage handlers so that no sig accidentally forgets to check.
		err = resourceExpirationEvaluator.RemoveUnavailableKinds(groupName, apiGroupInfo.Scheme, apiGroupInfo.VersionedResourcesStorageMap, s.APIResourceConfigSource)
		if err != nil {
			return err
		}
		if len(apiGroupInfo.VersionedResourcesStorageMap) == 0 {
			klog.V(1).Infof("Removing API group %v because it is time to stop serving it because it has no versions per APILifecycle.", groupName)
			continue
		}

		klog.V(1).Infof("Enabling API group %q.", groupName)

		if postHookProvider, ok := restStorageBuilder.(genericapiserver.PostStartHookProvider); ok {
			name, hook, err := postHookProvider.PostStartHook()
			if err != nil {
				return fmt.Errorf("error building PostStartHook: %w", err)
			}
			s.GenericAPIServer.AddPostStartHookOrDie(name, hook)
		}

		if len(groupName) == 0 {
			// the legacy group for core APIs is special that it is installed into /api via this special install method.
			if err := s.GenericAPIServer.InstallLegacyAPIGroup(genericapiserver.DefaultLegacyAPIPrefix, &apiGroupInfo); err != nil {
				return fmt.Errorf("error in registering legacy API: %w", err)
			}
		} else {
			// everything else goes to /apis
			nonLegacy = append(nonLegacy, &apiGroupInfo)
		}
	}

	if err := s.GenericAPIServer.InstallAPIGroups(nonLegacy...); err != nil {
		return fmt.Errorf("error in registering group versions: %w", err)
	}
	return nil
}
