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
	"k8s.io/client-go/tools/cache"
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
	PluginName = "CEL"
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
}

var _ WantsCELPolicyEvaluator = &celAdmissionPlugin{}
var _ admission.ValidationInterface = &celAdmissionPlugin{}

func NewPlugin() (*celAdmissionPlugin, error) {
	result := &celAdmissionPlugin{}
	return result, nil
}

func (c *celAdmissionPlugin) SetCELPolicyEvaluator(evaluator CELPolicyEvaluator) {
	c.evaluator = evaluator
}

// Once clientset and informer factory are provided, creates and starts the
// admission controller
func (c *celAdmissionPlugin) ValidateInitialization() error {
	if c.evaluator != nil {
		return nil
	}

	return errors.New("CELPolicyEvaluator not injected")
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

	deadlined, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	if !cache.WaitForNamedCacheSync("cel-admission-plugin", deadlined.Done(), c.evaluator.HasSynced) {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	return c.evaluator.Validate(ctx, a, o)
}
