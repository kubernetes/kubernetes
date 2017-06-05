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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/mux"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion"
	"k8s.io/kube-aggregator/pkg/controllers/autoregister"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/master/thirdparty"
)

func createAggregatorConfig(kubeAPIServerConfig genericapiserver.Config, commandOptions *options.ServerRunOptions) (*aggregatorapiserver.Config, error) {
	// make a shallow copy to let us twiddle a few things
	// most of the config actually remains the same.  We only need to mess with a couple items related to the particulars of the aggregator
	genericConfig := kubeAPIServerConfig
	genericConfig.FallThroughHandler = mux.NewPathRecorderMux()

	// the aggregator doesn't wire these up.  It just delegates them to the kubeapiserver
	genericConfig.EnableSwaggerUI = false
	genericConfig.OpenAPIConfig = nil
	genericConfig.SwaggerConfig = nil

	// copy the loopbackclientconfig.  We're going to change the contenttype back to json until we get protobuf serializations for it
	t := *kubeAPIServerConfig.LoopbackClientConfig
	genericConfig.LoopbackClientConfig = &t
	genericConfig.LoopbackClientConfig.ContentConfig.ContentType = ""

	// copy the etcd options so we don't mutate originals.
	etcdOptions := *commandOptions.Etcd
	etcdOptions.StorageConfig.Codec = aggregatorapiserver.Codecs.LegacyCodec(schema.GroupVersion{Group: "apiregistration.k8s.io", Version: "v1alpha1"})
	etcdOptions.StorageConfig.Copier = aggregatorapiserver.Scheme
	genericConfig.RESTOptionsGetter = &genericoptions.SimpleRestOptionsFactory{Options: etcdOptions}

	client, err := kubeclientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}

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
		GenericConfig:       &genericConfig,
		CoreAPIServerClient: client,
		ProxyClientCert:     certBytes,
		ProxyClientKey:      keyBytes,
	}

	return aggregatorConfig, nil

}

func createAggregatorServer(aggregatorConfig *aggregatorapiserver.Config, delegateAPIServer genericapiserver.DelegationTarget, sharedInformers informers.SharedInformerFactory, stopCh <-chan struct{}) (*aggregatorapiserver.APIAggregator, error) {
	aggregatorServer, err := aggregatorConfig.Complete().NewWithDelegate(delegateAPIServer, stopCh)
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
	tprRegistrationController := thirdparty.NewAutoRegistrationController(sharedInformers.Extensions().InternalVersion().ThirdPartyResources(), autoRegistrationController)

	aggregatorServer.GenericAPIServer.AddPostStartHook("kube-apiserver-autoregistration", func(context genericapiserver.PostStartHookContext) error {
		go autoRegistrationController.Run(5, stopCh)
		go tprRegistrationController.Run(5, stopCh)
		return nil
	})
	aggregatorServer.GenericAPIServer.AddHealthzChecks(healthz.NamedCheck("autoregister-completion", func(r *http.Request) error {
		items, err := aggregatorServer.APIRegistrationInformers.Apiregistration().InternalVersion().APIServices().Lister().List(labels.Everything())
		if err != nil {
			return err
		}

		missing := []apiregistration.APIService{}
		for _, apiService := range apiServices {
			found := false
			for _, item := range items {
				if item.Name == apiService.Name {
					found = true
					break
				}
			}

			if !found {
				missing = append(missing, *apiService)
			}
		}

		if len(missing) > 0 {
			return fmt.Errorf("missing APIService: %v", missing)
		}
		return nil
	}))

	return aggregatorServer, nil
}

func makeAPIService(gv schema.GroupVersion) *apiregistration.APIService {
	return &apiregistration.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: gv.Version + "." + gv.Group},
		Spec: apiregistration.APIServiceSpec{
			Group:    gv.Group,
			Version:  gv.Version,
			Priority: 100,
		},
	}
}

func apiServicesToRegister(delegateAPIServer genericapiserver.DelegationTarget, registration autoregister.AutoAPIServiceRegistration) []*apiregistration.APIService {
	apiServices := []*apiregistration.APIService{}

	for _, curr := range delegateAPIServer.ListedPaths() {
		if curr == "/api/v1" {
			apiService := makeAPIService(schema.GroupVersion{Group: "", Version: "v1"})
			registration.AddAPIServiceToSync(apiService)
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

		// TODO this is probably an indication that we need explicit and precise control over the discovery chain
		// but for now its a special case
		// apps has to come last for compatibility with 1.5 kubectl clients
		if apiService.Spec.Group == "apps" {
			apiService.Spec.Priority = 110
		}

		registration.AddAPIServiceToSync(apiService)
		apiServices = append(apiServices, apiService)
	}

	return apiServices
}
