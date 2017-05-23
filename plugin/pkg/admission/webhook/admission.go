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

// Package webhook checks a webhook for configured operation admission
package webhook

import (
	"errors"
	"fmt"
	"io"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/webhook"

	"k8s.io/kubernetes/pkg/api"
	admissionv1alpha1 "k8s.io/kubernetes/pkg/apis/admission/v1alpha1"

	// install the clientgo admissiony API for use with api registry
	_ "k8s.io/kubernetes/pkg/apis/admission/install"
)

var (
	groupVersions = []schema.GroupVersion{
		admissionv1alpha1.SchemeGroupVersion,
	}
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("GenericAdmissionWebhook", func(configFile io.Reader) (admission.Interface, error) {
		var gwhConfig struct {
			WebhookConfig GenericAdmissionWebhookConfig `json:"webhook"`
		}

		d := yaml.NewYAMLOrJSONDecoder(configFile, 4096)
		err := d.Decode(&gwhConfig)

		if err != nil {
			return nil, err
		}

		plugin, err := NewGenericAdmissionWebhook(&gwhConfig.WebhookConfig)

		if err != nil {
			return nil, err
		}

		return plugin, nil
	})
}

// NewGenericAdmissionWebhook returns a generic admission webhook plugin.
func NewGenericAdmissionWebhook(config *GenericAdmissionWebhookConfig) (admission.Interface, error) {
	err := normalizeConfig(config)

	if err != nil {
		return nil, err
	}

	gw, err := webhook.NewGenericWebhook(api.Registry, api.Codecs, config.KubeConfigFile, groupVersions, config.RetryBackoff)

	if err != nil {
		return nil, err
	}

	return &GenericAdmissionWebhook{
		Handler: admission.NewHandler(admission.Connect, admission.Create, admission.Delete, admission.Update),
		webhook: gw,
		rules:   config.Rules,
	}, nil
}

// GenericAdmissionWebhook is an implementation of admission.Interface.
type GenericAdmissionWebhook struct {
	*admission.Handler
	webhook *webhook.GenericWebhook
	rules   []Rule
}

// Admit makes an admission decision based on the request attributes.
func (a *GenericAdmissionWebhook) Admit(attr admission.Attributes) (err error) {
	var matched *Rule

	// Process all declared rules to attempt to find a match
	for i, rule := range a.rules {
		if Matches(rule, attr) {
			glog.V(2).Infof("rule at index %d matched request", i)
			matched = &a.rules[i]
			break
		}
	}

	if matched == nil {
		glog.V(2).Infof("rule explicitly allowed the request: no rule matched the admission request")
		return nil
	}

	// The matched rule skips processing this request
	if matched.Type == Skip {
		glog.V(2).Infof("rule explicitly allowed the request")
		return nil
	}

	// Make the webhook request
	request := admissionv1alpha1.NewAdmissionReview(attr)
	response := a.webhook.RestClient.Post().Body(&request).Do()

	// Handle webhook response
	if err := response.Error(); err != nil {
		return a.handleError(attr, matched.FailAction, err)
	}

	var statusCode int
	if response.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
		return a.handleError(attr, matched.FailAction, fmt.Errorf("error contacting webhook: %d", statusCode))
	}

	if err := response.Into(&request); err != nil {
		return a.handleError(attr, matched.FailAction, err)
	}

	if !request.Status.Allowed {
		if request.Status.Result != nil && len(request.Status.Result.Reason) > 0 {
			return a.handleError(attr, Deny, fmt.Errorf("webhook backend denied the request: %s", request.Status.Result.Reason))
		}
		return a.handleError(attr, Deny, errors.New("webhook backend denied the request"))
	}

	// The webhook admission controller DOES NOT allow mutation of the admission request so nothing else is required

	return nil
}

func (a *GenericAdmissionWebhook) handleError(attr admission.Attributes, allowIfErr FailAction, err error) error {
	if err != nil {
		glog.V(2).Infof("error contacting webhook backend: %s", err)
		if allowIfErr != Allow {
			glog.V(2).Infof("resource not allowed due to webhook backend failure: %s", err)
			return admission.NewForbidden(attr, err)
		}
		glog.V(2).Infof("resource allowed in spite of webhook backend failure")
	}

	return nil
}

// TODO: Allow configuring the serialization strategy
