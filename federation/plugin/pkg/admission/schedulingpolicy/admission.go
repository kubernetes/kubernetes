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

// Package schedulingpolicy implements a webhook that queries an external API
// to obtain scheduling decisions for Federated sources.
package schedulingpolicy

import (
	"fmt"
	"io"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/ref"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

const (
	pluginName               = "SchedulingPolicy"
	configKey                = "schedulingPolicy"
	policyConfigMapNamespace = "kube-federation-scheduling-policy"

	// Default backoff delay for policy engine query retries. The actual
	// backoff implementation is handled by k8s.io/apiserver/pkg/util/webhook.
	// If the admission controller config file does not specify a backoff, this
	// one is used.
	defaultRetryBackoff = time.Millisecond * 100
)

type admissionConfig struct {
	Kubeconfig   string        `json:"kubeconfig"`
	RetryBackoff time.Duration `json:"retryBackoff"`
}

type admissionController struct {
	*admission.Handler
	policyEngineClient       *rest.RESTClient            // client to communicate with policy engine
	policyEngineRetryBackoff time.Duration               // backoff for policy engine queries
	client                   internalclientset.Interface // client to communicate with federation-apiserver
}

// Register registers the plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(pluginName, func(file io.Reader) (admission.Interface, error) {
		return newAdmissionController(file)
	})
}

func newAdmissionController(file io.Reader) (*admissionController, error) {
	config, err := loadConfig(file)
	if err != nil {
		return nil, err
	}

	policyEngineClient, err := loadRestClient(config.Kubeconfig)
	if err != nil {
		return nil, err
	}

	c := &admissionController{
		Handler:                  admission.NewHandler(admission.Create, admission.Update),
		policyEngineClient:       policyEngineClient,
		policyEngineRetryBackoff: config.RetryBackoff,
	}

	return c, nil
}

func (c *admissionController) Validate() error {
	if c.client == nil {
		return fmt.Errorf("%s requires a client", pluginName)
	}
	return nil
}

func (c *admissionController) SetInternalKubeClientSet(client internalclientset.Interface) {
	c.client = client
}

func (c *admissionController) Admit(a admission.Attributes) (err error) {
	exists, err := c.policyExists()
	if err != nil {
		return c.handleError(a, err)
	}

	if !exists {
		return nil
	}

	obj := a.GetObject()
	decision, err := newPolicyEngineQuery(c.policyEngineClient, c.policyEngineRetryBackoff, obj, a.GetKind()).Do()

	if err != nil {
		return c.handleError(a, err)
	}

	if err := decision.Error(); err != nil {
		return c.handleError(a, err)
	}

	mergeAnnotations(obj, decision.Annotations)

	return nil
}

func (c *admissionController) handleError(a admission.Attributes, err error) error {

	c.publishEvent(a, err.Error())

	return admission.NewForbidden(a, err)
}

func (c *admissionController) publishEvent(a admission.Attributes, msg string) {

	obj := a.GetObject()

	ref, err := ref.GetReference(api.Scheme, obj)
	if err != nil {
		runtime.HandleError(err)
		return
	}

	event := &api.Event{
		InvolvedObject: *ref,
		Message:        msg,
		Source: api.EventSource{
			Component: fmt.Sprintf("schedulingpolicy"),
		},
		Type: "Warning",
	}

	if _, err := c.client.Core().Events(a.GetNamespace()).Create(event); err != nil {
		runtime.HandleError(err)
		return
	}
}

func (c *admissionController) policyExists() (bool, error) {
	lst, err := c.client.Core().ConfigMaps(policyConfigMapNamespace).List(metav1.ListOptions{})
	if err != nil {
		return true, err
	}
	return len(lst.Items) > 0, nil
}

func loadConfig(file io.Reader) (*admissionConfig, error) {
	var cfg admissionConfig
	if file == nil {
		return nil, fmt.Errorf("--admission-control-config-file not specified or invalid")
	}

	if err := yaml.NewYAMLOrJSONDecoder(file, 4096).Decode(&cfg); err != nil {
		return nil, err
	}

	if len(cfg.Kubeconfig) == 0 {
		return nil, fmt.Errorf("kubeconfig path must not be empty")
	}

	if cfg.RetryBackoff == 0 {
		cfg.RetryBackoff = defaultRetryBackoff
	} else {
		// Scale up value from config (which is unmarshalled as ns).
		cfg.RetryBackoff *= time.Millisecond
	}

	if cfg.RetryBackoff.Nanoseconds() < 0 {
		return nil, fmt.Errorf("retryBackoff must not be negative")
	}

	return &cfg, nil
}

func loadRestClient(kubeConfigFile string) (*rest.RESTClient, error) {

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	loadingRules.ExplicitPath = kubeConfigFile
	loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

	clientConfig, err := loader.ClientConfig()
	if err != nil {
		return nil, err
	}

	clientConfig.ContentConfig.NegotiatedSerializer = dynamic.ContentConfig().NegotiatedSerializer

	restClient, err := rest.UnversionedRESTClientFor(clientConfig)
	if err != nil {
		return nil, err
	}

	return restClient, nil
}
