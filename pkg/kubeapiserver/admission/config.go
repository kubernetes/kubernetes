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

package admission

import (
	"io/ioutil"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/trace"

	utilwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/cel/matching"
	webhookinit "k8s.io/apiserver/pkg/admission/plugin/webhook/initializer"
	"k8s.io/apiserver/pkg/features"
	genericapiserver "k8s.io/apiserver/pkg/server"
	egressselector "k8s.io/apiserver/pkg/server/egressselector"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	externalinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
)

// Config holds the configuration needed to for initialize the admission plugins
type Config struct {
	CloudConfigFile      string
	LoopbackClientConfig *rest.Config
	ExternalInformers    externalinformers.SharedInformerFactory
}

// New sets up the plugins and admission start hooks needed for admission
func (c *Config) New(proxyTransport *http.Transport, egressSelector *egressselector.EgressSelector, serviceResolver webhook.ServiceResolver, tp trace.TracerProvider) ([]admission.PluginInitializer, genericapiserver.PostStartHookFunc, error) {
	webhookAuthResolverWrapper := webhook.NewDefaultAuthenticationInfoResolverWrapper(proxyTransport, egressSelector, c.LoopbackClientConfig, tp)
	webhookPluginInitializer := webhookinit.NewPluginInitializer(webhookAuthResolverWrapper, serviceResolver)

	var cloudConfig []byte
	if c.CloudConfigFile != "" {
		var err error
		cloudConfig, err = ioutil.ReadFile(c.CloudConfigFile)
		if err != nil {
			klog.Fatalf("Error reading from cloud configuration file %s: %#v", c.CloudConfigFile, err)
		}
	}
	clientset, err := kubernetes.NewForConfig(c.LoopbackClientConfig)
	if err != nil {
		return nil, nil, err
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientset.Discovery())
	discoveryRESTMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	kubePluginInitializer := NewPluginInitializer(
		cloudConfig,
		discoveryRESTMapper,
		quotainstall.NewQuotaConfigurationForAdmission(),
	)

	admissionPostStartHook := func(context genericapiserver.PostStartHookContext) error {
		discoveryRESTMapper.Reset()
		go utilwait.Until(discoveryRESTMapper.Reset, 30*time.Second, context.StopCh)
		return nil
	}

	initializers := []admission.PluginInitializer{webhookPluginInitializer, kubePluginInitializer}

	if utilfeature.DefaultFeatureGate.Enabled(features.CELValidatingAdmission) {
		dynamicClient, err := dynamic.NewForConfig(c.LoopbackClientConfig)
		if err != nil {
			return nil, nil, err
		}

		// Create CEL admission controller
		var celAdmissionController = cel.NewAdmissionController(
			c.ExternalInformers.Admissionregistration().V1alpha1().ValidatingAdmissionPolicies().Informer(),
			c.ExternalInformers.Admissionregistration().V1alpha1().ValidatingAdmissionPolicyBindings().Informer(),
			&cel.CELValidatorCompiler{
				Matcher: matching.NewMatcher()},
			discoveryRESTMapper,
			dynamicClient,
		)
		celAdmissionPluginInitializer := cel.NewPluginInitializer(celAdmissionController)
		initializers = append(initializers, celAdmissionPluginInitializer)

		hookDelegate := admissionPostStartHook
		admissionPostStartHook = func(context genericapiserver.PostStartHookContext) error {
			go celAdmissionController.Run(context.StopCh)
			return hookDelegate(context)
		}
	}

	return initializers, admissionPostStartHook, nil

}
