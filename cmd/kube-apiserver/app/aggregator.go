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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"

	"k8s.io/klog"

	apiextensionsinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubeexternalinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion"
	informers "k8s.io/kube-aggregator/pkg/client/informers/internalversion/apiregistration/internalversion"
	"k8s.io/kube-aggregator/pkg/controllers/autoregister"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/master/controller/crdregistration"
)

func createAggregatorConfig(
	kubeAPIServerConfig genericapiserver.Config,
	commandOptions *options.ServerRunOptions,
	externalInformers kubeexternalinformers.SharedInformerFactory,
	serviceResolver aggregatorapiserver.ServiceResolver,
	proxyTransport *http.Transport,
	pluginInitializers []admission.PluginInitializer,
) (*aggregatorapiserver.Config, error) {
	// make a shallow copy to let us twiddle a few things
	// most of the config actually remains the same.  We only need to mess with a couple items related to the particulars of the aggregator
	genericConfig := kubeAPIServerConfig

	// override genericConfig.AdmissionControl with kube-aggregator's scheme,
	// because aggregator apiserver should use its own scheme to convert its own resources.
	commandOptions.Admission.ApplyTo(
		&genericConfig,
		externalInformers,
		genericConfig.LoopbackClientConfig,
		aggregatorscheme.Scheme,
		pluginInitializers...)

	// the aggregator doesn't wire these up.  It just delegates them to the kubeapiserver
	genericConfig.EnableSwaggerUI = false
	genericConfig.SwaggerConfig = nil

	// copy the etcd options so we don't mutate originals.
	etcdOptions := *commandOptions.Etcd
	etcdOptions.StorageConfig.Paging = utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	etcdOptions.StorageConfig.Codec = aggregatorscheme.Codecs.LegacyCodec(v1beta1.SchemeGroupVersion, v1.SchemeGroupVersion)
	genericConfig.RESTOptionsGetter = &genericoptions.SimpleRestOptionsFactory{Options: etcdOptions}

	// override MergedResourceConfig with aggregator defaults and registry
	if err := commandOptions.APIEnablement.ApplyTo(
		&genericConfig,
		aggregatorapiserver.DefaultAPIResourceConfigSource(),
		aggregatorscheme.Scheme); err != nil {
		return nil, err
	}

	var err error
	var certBytes, keyBytes []byte
	if len(commandOptions.ProxyClientCertFile) > 0 && len(commandOptions.ProxyClientKeyFile) > 0 {
		certBytes, err = ioutil.ReadFile(commandOptions.ProxyClientCertFile)
		if err != nil {
			return nil, err
		}
		keyBytes, err = ioutil.ReadFile(commandOptions.ProxyClientKeyFile)
		if err != nil {
			return nil, err
		}
	}

	aggregatorConfig := &aggregatorapiserver.Config{
		GenericConfig: &genericapiserver.RecommendedConfig{
			Config:                genericConfig,
			SharedInformerFactory: externalInformers,
		},
		ExtraConfig: aggregatorapiserver.ExtraConfig{
			ProxyClientCert: certBytes,
			ProxyClientKey:  keyBytes,
			ServiceResolver: serviceResolver,
			ProxyTransport:  proxyTransport,
		},
	}

	return aggregatorConfig, nil
}

func createAggregatorServer(aggregatorConfig *aggregatorapiserver.Config, delegateAPIServer genericapiserver.DelegationTarget, apiExtensionInformers apiextensionsinformers.SharedInformerFactory) (*aggregatorapiserver.APIAggregator, error) {
	aggregatorServer, err := aggregatorConfig.Complete().NewWithDelegate(delegateAPIServer)
	if err != nil {
		return nil, err
	}

	// create controllers for auto-registration
	apiRegistrationClient, err := apiregistrationclient.NewForConfig(aggregatorConfig.GenericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	autoRegistrationController := autoregister.NewAutoRegisterController(aggregatorServer.APIRegistrationInformers.Apiregistration().InternalVersion().APIServices(), apiRegistrationClient)
	apiServices := apiServicesToRegister(delegateAPIServer, autoRegistrationController)
	crdRegistrationController := crdregistration.NewAutoRegistrationController(
		apiExtensionInformers.Apiextensions().InternalVersion().CustomResourceDefinitions(),
		autoRegistrationController)

	aggregatorServer.GenericAPIServer.AddPostStartHook("kube-apiserver-autoregistration", func(context genericapiserver.PostStartHookContext) error {
		go crdRegistrationController.Run(5, context.StopCh)
		go func() {
			// let the CRD controller process the initial set of CRDs before starting the autoregistration controller.
			// this prevents the autoregistration controller's initial sync from deleting APIServices for CRDs that still exist.
			// we only need to do this if CRDs are enabled on this server.  We can't use discovery because we are the source for discovery.
			if aggregatorConfig.GenericConfig.MergedResourceConfig.AnyVersionForGroupEnabled("apiextensions.k8s.io") {
				crdRegistrationController.WaitForInitialSync()
			}
			autoRegistrationController.Run(5, context.StopCh)
		}()
		return nil
	})

	aggregatorServer.GenericAPIServer.AddHealthzChecks(
		makeAPIServiceAvailableHealthzCheck(
			"autoregister-completion",
			apiServices,
			aggregatorServer.APIRegistrationInformers.Apiregistration().InternalVersion().APIServices(),
		),
	)

	return aggregatorServer, nil
}

func makeAPIService(gv schema.GroupVersion) *apiregistration.APIService {
	apiServicePriority, ok := apiVersionPriorities[gv]
	if !ok {
		// if we aren't found, then we shouldn't register ourselves because it could result in a CRD group version
		// being permanently stuck in the APIServices list.
		klog.Infof("Skipping APIService creation for %v", gv)
		return nil
	}
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: gv.Version + "." + gv.Group},
		Spec: apiregistration.APIServiceSpec{
			Group:                gv.Group,
			Version:              gv.Version,
			GroupPriorityMinimum: apiServicePriority.group,
			VersionPriority:      apiServicePriority.version,
		},
	}
}

// makeAPIServiceAvailableHealthzCheck returns a healthz check that returns healthy
// once all of the specified services have been observed to be available at least once.
func makeAPIServiceAvailableHealthzCheck(name string, apiServices []*apiregistration.APIService, apiServiceInformer informers.APIServiceInformer) healthz.HealthzChecker {
	// Track the auto-registered API services that have not been observed to be available yet
	pendingServiceNamesLock := &sync.RWMutex{}
	pendingServiceNames := sets.NewString()
	for _, service := range apiServices {
		pendingServiceNames.Insert(service.Name)
	}

	// When an APIService in the list is seen as available, remove it from the pending list
	handleAPIServiceChange := func(service *apiregistration.APIService) {
		pendingServiceNamesLock.Lock()
		defer pendingServiceNamesLock.Unlock()
		if !pendingServiceNames.Has(service.Name) {
			return
		}
		if apiregistration.IsAPIServiceConditionTrue(service, apiregistration.Available) {
			pendingServiceNames.Delete(service.Name)
		}
	}

	// Watch add/update events for APIServices
	apiServiceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { handleAPIServiceChange(obj.(*apiregistration.APIService)) },
		UpdateFunc: func(old, new interface{}) { handleAPIServiceChange(new.(*apiregistration.APIService)) },
	})

	// Don't return healthy until the pending list is empty
	return healthz.NamedCheck(name, func(r *http.Request) error {
		pendingServiceNamesLock.RLock()
		defer pendingServiceNamesLock.RUnlock()
		if pendingServiceNames.Len() > 0 {
			return fmt.Errorf("missing APIService: %v", pendingServiceNames.List())
		}
		return nil
	})
}

// priority defines group priority that is used in discovery. This controls
// group position in the kubectl output.
type priority struct {
	// group indicates the order of the group relative to other groups.
	group int32
	// version indicates the relative order of the version inside of its group.
	version int32
}

// The proper way to resolve this letting the aggregator know the desired group and version-within-group order of the underlying servers
// is to refactor the genericapiserver.DelegationTarget to include a list of priorities based on which APIs were installed.
// This requires the APIGroupInfo struct to evolve and include the concept of priorities and to avoid mistakes, the core storage map there needs to be updated.
// That ripples out every bit as far as you'd expect, so for 1.7 we'll include the list here instead of being built up during storage.
var apiVersionPriorities = map[schema.GroupVersion]priority{
	{Group: "", Version: "v1"}: {group: 18000, version: 1},
	// extensions is above the rest for CLI compatibility, though the level of unqualified resource compatibility we
	// can reasonably expect seems questionable.
	{Group: "extensions", Version: "v1beta1"}: {group: 17900, version: 1},
	// to my knowledge, nothing below here collides
	{Group: "apps", Version: "v1beta1"}:                          {group: 17800, version: 1},
	{Group: "apps", Version: "v1beta2"}:                          {group: 17800, version: 9},
	{Group: "apps", Version: "v1"}:                               {group: 17800, version: 15},
	{Group: "events.k8s.io", Version: "v1beta1"}:                 {group: 17750, version: 5},
	{Group: "authentication.k8s.io", Version: "v1"}:              {group: 17700, version: 15},
	{Group: "authentication.k8s.io", Version: "v1beta1"}:         {group: 17700, version: 9},
	{Group: "authorization.k8s.io", Version: "v1"}:               {group: 17600, version: 15},
	{Group: "authorization.k8s.io", Version: "v1beta1"}:          {group: 17600, version: 9},
	{Group: "autoscaling", Version: "v1"}:                        {group: 17500, version: 15},
	{Group: "autoscaling", Version: "v2beta1"}:                   {group: 17500, version: 9},
	{Group: "autoscaling", Version: "v2beta2"}:                   {group: 17500, version: 1},
	{Group: "batch", Version: "v1"}:                              {group: 17400, version: 15},
	{Group: "batch", Version: "v1beta1"}:                         {group: 17400, version: 9},
	{Group: "batch", Version: "v2alpha1"}:                        {group: 17400, version: 9},
	{Group: "certificates.k8s.io", Version: "v1beta1"}:           {group: 17300, version: 9},
	{Group: "networking.k8s.io", Version: "v1"}:                  {group: 17200, version: 15},
	{Group: "policy", Version: "v1beta1"}:                        {group: 17100, version: 9},
	{Group: "rbac.authorization.k8s.io", Version: "v1"}:          {group: 17000, version: 15},
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1"}:     {group: 17000, version: 12},
	{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"}:    {group: 17000, version: 9},
	{Group: "settings.k8s.io", Version: "v1alpha1"}:              {group: 16900, version: 9},
	{Group: "storage.k8s.io", Version: "v1"}:                     {group: 16800, version: 15},
	{Group: "storage.k8s.io", Version: "v1beta1"}:                {group: 16800, version: 9},
	{Group: "storage.k8s.io", Version: "v1alpha1"}:               {group: 16800, version: 1},
	{Group: "apiextensions.k8s.io", Version: "v1beta1"}:          {group: 16700, version: 9},
	{Group: "admissionregistration.k8s.io", Version: "v1"}:       {group: 16700, version: 15},
	{Group: "admissionregistration.k8s.io", Version: "v1beta1"}:  {group: 16700, version: 12},
	{Group: "admissionregistration.k8s.io", Version: "v1alpha1"}: {group: 16700, version: 9},
	{Group: "scheduling.k8s.io", Version: "v1beta1"}:             {group: 16600, version: 12},
	{Group: "scheduling.k8s.io", Version: "v1alpha1"}:            {group: 16600, version: 9},
	{Group: "coordination.k8s.io", Version: "v1beta1"}:           {group: 16500, version: 9},
	{Group: "auditregistration.k8s.io", Version: "v1alpha1"}:     {group: 16400, version: 1},
	// Append a new group to the end of the list if unsure.
	// You can use min(existing group)-100 as the initial value for a group.
	// Version can be set to 9 (to have space around) for a new group.
}

func apiServicesToRegister(delegateAPIServer genericapiserver.DelegationTarget, registration autoregister.AutoAPIServiceRegistration) []*apiregistration.APIService {
	apiServices := []*apiregistration.APIService{}

	for _, curr := range delegateAPIServer.ListedPaths() {
		if curr == "/api/v1" {
			apiService := makeAPIService(schema.GroupVersion{Group: "", Version: "v1"})
			registration.AddAPIServiceToSyncOnStart(apiService)
			apiServices = append(apiServices, apiService)
			continue
		}

		if !strings.HasPrefix(curr, "/apis/") {
			continue
		}
		// this comes back in a list that looks like /apis/rbac.authorization.k8s.io/v1alpha1
		tokens := strings.Split(curr, "/")
		if len(tokens) != 4 {
			continue
		}

		apiService := makeAPIService(schema.GroupVersion{Group: tokens[2], Version: tokens[3]})
		if apiService == nil {
			continue
		}
		registration.AddAPIServiceToSyncOnStart(apiService)
		apiServices = append(apiServices, apiService)
	}

	return apiServices
}
