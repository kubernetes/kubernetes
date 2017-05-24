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

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

	// Lower and upper bounds on policy engine query retry delays. The actual
	// backoff implementation is handled by k8s.io/apiserver/pkg/util/webhook.
	// The defaultRetryBackoff is used if the admission controller
	// configuration does not specify one.
	minRetryBackoff     = time.Millisecond
	maxRetryBackoff     = time.Second * 60
	defaultRetryBackoff = time.Millisecond * 100
)

type admissionConfig struct {
	Kubeconfig                   string        `json:"kubeconfig"`
	RetryBackoff                 time.Duration `json:"retryBackoff"`
	FailOpenIfPolicyDoesNotExist bool          `json:"failOpenIfPolicyDoesNotExist"`
}

type admissionController struct {
	*admission.Handler
	policyEngineClient       *rest.RESTClient            // client to communicate with policy engine
	policyEngineRetryBackoff time.Duration               // backoff for policy engine queries
	client                   internalclientset.Interface // client to communicate with federation-apiserver

	// controls whether policy engine is queried when policy has not been
	// created in policyConfigMapNamespace. By default, policy engine IS NOT
	// queried unless policy is known to exist. If this flag is true, the
	// policy engine IS queried however the admission controller will fail
	// open.
	failOpenIfPolicyDoesNotExist bool
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
		Handler:                      admission.NewHandler(admission.Create, admission.Update),
		policyEngineClient:           policyEngineClient,
		policyEngineRetryBackoff:     config.RetryBackoff,
		failOpenIfPolicyDoesNotExist: config.FailOpenIfPolicyDoesNotExist,
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
		return c.handleError(a, true, err)
	}

	failClosed := true
	if !exists {
		if !c.failOpenIfPolicyDoesNotExist {
			return nil
		}
		failClosed = false
	}

	obj := a.GetObject()
	decision, err := newPolicyEngineQuery(c.policyEngineClient, c.policyEngineRetryBackoff, obj, a.GetKind()).Do()

	if err != nil {
		return c.handleError(a, failClosed, err)
	}

	if err := decision.Error(); err != nil {
		return c.handleError(a, true, err)
	}

	mergeAnnotations(obj, decision.Annotations)

	return nil
}

func (c *admissionController) handleError(a admission.Attributes, failClosed bool, err error) error {
	if !failClosed {
		glog.V(4).Infof("an error occurred but request will be allowed (%v namespace does not contain any ConfigMap resources): %v.", policyConfigMapNamespace, err.Error())
		return nil
	}

	c.publishEvent(a, err.Error())

	return admission.NewForbidden(a, err)
}

func (c *admissionController) publishEvent(a admission.Attributes, msg string) {

	obj := a.GetObject()

	ref, err := ref.GetReference(api.Scheme, obj)
	if err != nil {
		glog.Errorf("get reference for %v (kind: %v) failed: %v", a.GetName(), a.GetKind().Kind, err.Error())
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
		glog.Errorf("create event for %v (kind: %v) failed: %v", a.GetName(), a.GetKind().Kind, err.Error())
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
		return nil, fmt.Errorf("kubeconfig must specify path")
	}

	if cfg.RetryBackoff == 0 {
		cfg.RetryBackoff = defaultRetryBackoff
	} else {
		// Scale up value from config (which is unmarshalled as ns).
		cfg.RetryBackoff *= time.Millisecond
		if cfg.RetryBackoff < minRetryBackoff || cfg.RetryBackoff > maxRetryBackoff {
			return nil, fmt.Errorf("retryBackoff must be between %v and %v but got %v", minRetryBackoff, maxRetryBackoff, cfg.RetryBackoff)
		}
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
