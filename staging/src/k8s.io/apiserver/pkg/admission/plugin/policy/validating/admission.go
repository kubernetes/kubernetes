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

package validating

import (
	"context"
	"errors"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/dynamic"
	"k8s.io/component-base/featuregate"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
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
	*admission.Handler
	evaluator CELPolicyEvaluator

	inspectedFeatureGates bool
	enabled               bool

	// Injected Dependencies
	informerFactory informers.SharedInformerFactory
	client          kubernetes.Interface
	restMapper      meta.RESTMapper
	dynamicClient   dynamic.Interface
	stopCh          <-chan struct{}
	authorizer      authorizer.Authorizer
}

var _ initializer.WantsExternalKubeInformerFactory = &celAdmissionPlugin{}
var _ initializer.WantsExternalKubeClientSet = &celAdmissionPlugin{}
var _ initializer.WantsRESTMapper = &celAdmissionPlugin{}
var _ initializer.WantsDynamicClient = &celAdmissionPlugin{}
var _ initializer.WantsDrainedNotification = &celAdmissionPlugin{}
var _ initializer.WantsAuthorizer = &celAdmissionPlugin{}
var _ admission.InitializationValidator = &celAdmissionPlugin{}
var _ admission.ValidationInterface = &celAdmissionPlugin{}

func NewPlugin() (admission.Interface, error) {
	return &celAdmissionPlugin{
		Handler: admission.NewHandler(admission.Connect, admission.Create, admission.Delete, admission.Update),
	}, nil
}

func (c *celAdmissionPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	c.informerFactory = f
}

func (c *celAdmissionPlugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	c.client = client
}

func (c *celAdmissionPlugin) SetRESTMapper(mapper meta.RESTMapper) {
	c.restMapper = mapper
}

func (c *celAdmissionPlugin) SetDynamicClient(client dynamic.Interface) {
	c.dynamicClient = client
}

func (c *celAdmissionPlugin) SetDrainedNotification(stopCh <-chan struct{}) {
	c.stopCh = stopCh
}

func (c *celAdmissionPlugin) SetAuthorizer(authorizer authorizer.Authorizer) {
	c.authorizer = authorizer
}
func (c *celAdmissionPlugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	if featureGates.Enabled(features.ValidatingAdmissionPolicy) {
		c.enabled = true
	}
	c.inspectedFeatureGates = true
}

// ValidateInitialization - once clientset and informer factory are provided, creates and starts the admission controller
func (c *celAdmissionPlugin) ValidateInitialization() error {
	if !c.inspectedFeatureGates {
		return fmt.Errorf("%s did not see feature gates", PluginName)
	}
	if !c.enabled {
		return nil
	}
	if c.informerFactory == nil {
		return errors.New("missing informer factory")
	}
	if c.client == nil {
		return errors.New("missing kubernetes client")
	}
	if c.restMapper == nil {
		return errors.New("missing rest mapper")
	}
	if c.dynamicClient == nil {
		return errors.New("missing dynamic client")
	}
	if c.stopCh == nil {
		return errors.New("missing stop channel")
	}
	if c.authorizer == nil {
		return errors.New("missing authorizer")
	}
	c.evaluator = NewAdmissionController(c.informerFactory, c.client, c.restMapper, c.dynamicClient, c.authorizer)
	if err := c.evaluator.ValidateInitialization(); err != nil {
		return err
	}

	c.SetReadyFunc(c.evaluator.HasSynced)
	go c.evaluator.Run(c.stopCh)
	return nil
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

	// isPolicyResource determines if an admission.Attributes object is describing
	// the admission of a ValidatingAdmissionPolicy or a ValidatingAdmissionPolicyBinding
	if isPolicyResource(a) {
		return
	}

	if !c.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	return c.evaluator.Validate(ctx, a, o)
}

func isPolicyResource(attr admission.Attributes) bool {
	gvk := attr.GetResource()
	if gvk.Group == "admissionregistration.k8s.io" {
		if gvk.Resource == "validatingadmissionpolicies" || gvk.Resource == "validatingadmissionpolicybindings" {
			return true
		}
	}
	return false
}
