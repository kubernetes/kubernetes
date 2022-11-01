/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
)

////////////////////////////////////////////////////////////////////////////////
// Plugin Definition
////////////////////////////////////////////////////////////////////////////////

// Definition for CEL admission plugin. This is the entry point into the
// CEL admission control system.
//
// Each plugin is asked to validate every object update.

const (
	// PluginName indicates the name of admission plug-in
	PluginName = "ValidatingAdmissionPolicy"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin()
	})
}

////////////////////////////////////////////////////////////////////////////////
// Plugin Initialization & Dependency Injection
////////////////////////////////////////////////////////////////////////////////

type celAdmissionPlugin struct {
	evaluator CELPolicyEvaluator
	enabled   bool
}

var _ WantsCELPolicyEvaluator = &celAdmissionPlugin{}
var _ admission.InitializationValidator = &celAdmissionPlugin{}
var _ admission.ValidationInterface = &celAdmissionPlugin{}

func NewPlugin() (admission.Interface, error) {
	result := &celAdmissionPlugin{}
	return result, nil
}

func (c *celAdmissionPlugin) SetCELPolicyEvaluator(evaluator CELPolicyEvaluator) {
	c.evaluator = evaluator
}

// Once clientset and informer factory are provided, creates and starts the admission controller
func (c *celAdmissionPlugin) ValidateInitialization() error {
	if !c.enabled {
		return nil
	}
	if c.evaluator == nil {
		return errors.New("CELPolicyEvaluator not injected")
	}
	return nil
}

func (c *celAdmissionPlugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	if featureGates.Enabled(features.CELValidatingAdmission) {
		c.enabled = true
	}
}

////////////////////////////////////////////////////////////////////////////////
// admission.ValidationInterface
////////////////////////////////////////////////////////////////////////////////

func (c *celAdmissionPlugin) Handles(operation admission.Operation) bool {
	return true
}

func (c *celAdmissionPlugin) Validate(
	ctx context.Context,
	a admission.Attributes,
	o admission.ObjectInterfaces,
) (err error) {
	if !c.enabled {
		return nil
	}

	deadlined, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	if !cache.WaitForNamedCacheSync("cel-admission-plugin", deadlined.Done(), c.evaluator.HasSynced) {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	// isPolicyResource determines if an admission.Attributes object is describing
	// the admission of a ValidatingAdmissionPolicy or a ValidatingAdmissionPolicyBinding
	if isPolicyResource(a) {
		return
	}

	return c.evaluator.Validate(ctx, a, o)
}

func isPolicyResource(attr admission.Attributes) bool {
	gvk := attr.GetKind()
	if gvk.Group == "admissionregistration.k8s.io" {
		if gvk.Kind == "ValidatingAdmissionPolicy" || gvk.Kind == "ValidatingAdmissionPolicyBinding" {
			return true
		}
	}
	return false
}
