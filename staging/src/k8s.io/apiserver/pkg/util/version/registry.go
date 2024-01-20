/*
Copyright 2024 The Kubernetes Authors.

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

package version

import (
	"fmt"
	"sync"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/featuregate"
)

var DefaultComponentGlobalsRegistry ComponentGlobalsRegistry = NewComponentGlobalsRegistry()

const (
	ComponentGenericAPIServer = "k8s.io/apiserver"
)

// ComponentGlobals stores the global variables for a component for easy access.
type ComponentGlobals struct {
	effectiveVersion MutableEffectiveVersion
	featureGate      featuregate.MutableVersionedFeatureGate
}

type ComponentGlobalsRegistry interface {
	// EffectiveVersionFor returns the EffectiveVersion registered under the component.
	// Returns nil if the component is not registered.
	EffectiveVersionFor(component string) EffectiveVersion
	// FeatureGateFor returns the FeatureGate registered under the component.
	// Returns nil if the component is not registered.
	FeatureGateFor(component string) featuregate.FeatureGate
	// Register registers the EffectiveVersion and FeatureGate for a component.
	// Overrides existing ComponentGlobals if it is already in the registry if override is true,
	// otherwise returns error if the component is already registered.
	Register(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate, override bool) error
	// ComponentGlobalsOrRegister would return the registered global variables for the component if it already exists in the registry.
	// Otherwise, the provided variables would be registered under the component, and the same variables would be returned.
	ComponentGlobalsOrRegister(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate) (MutableEffectiveVersion, featuregate.MutableVersionedFeatureGate)
	// SetAllComponents sets the emulation version for other global variables for all components registered.
	SetAllComponents() error
	// SetAllComponents calls the Validate() function for all the global variables for all components registered.
	ValidateAllComponents() []error
}

type componentGlobalsRegistry struct {
	componentGlobals map[string]ComponentGlobals
	mutex            sync.RWMutex
}

func NewComponentGlobalsRegistry() ComponentGlobalsRegistry {
	return &componentGlobalsRegistry{componentGlobals: map[string]ComponentGlobals{}}
}

func (r *componentGlobalsRegistry) EffectiveVersionFor(component string) EffectiveVersion {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	globals, ok := r.componentGlobals[component]
	if !ok {
		return nil
	}
	return globals.effectiveVersion
}

func (r *componentGlobalsRegistry) FeatureGateFor(component string) featuregate.FeatureGate {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	globals, ok := r.componentGlobals[component]
	if !ok {
		return nil
	}
	return globals.featureGate
}

func (r *componentGlobalsRegistry) unsafeRegister(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate, override bool) error {
	if _, ok := r.componentGlobals[component]; ok && !override {
		return fmt.Errorf("component globals of %s already registered", component)
	}
	if featureGate != nil {
		featureGate.DeferErrorsToValidation(true)
	}
	c := ComponentGlobals{effectiveVersion: effectiveVersion, featureGate: featureGate}
	r.componentGlobals[component] = c
	return nil
}

func (r *componentGlobalsRegistry) Register(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate, override bool) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	return r.unsafeRegister(component, effectiveVersion, featureGate, override)
}

func (r *componentGlobalsRegistry) ComponentGlobalsOrRegister(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate) (MutableEffectiveVersion, featuregate.MutableVersionedFeatureGate) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	globals, ok := r.componentGlobals[component]
	if ok {
		return globals.effectiveVersion, globals.featureGate
	}
	utilruntime.Must(r.unsafeRegister(component, effectiveVersion, featureGate, false))
	return effectiveVersion, featureGate
}

func (r *componentGlobalsRegistry) SetAllComponents() error {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	for _, globals := range r.componentGlobals {
		if globals.featureGate == nil {
			continue
		}
		if err := globals.featureGate.SetEmulationVersion(globals.effectiveVersion.EmulationVersion()); err != nil {
			return err
		}
	}
	return nil
}

func (r *componentGlobalsRegistry) ValidateAllComponents() []error {
	var errs []error
	r.mutex.Lock()
	defer r.mutex.Unlock()
	for _, globals := range r.componentGlobals {
		errs = append(errs, globals.effectiveVersion.Validate()...)
		if globals.featureGate != nil {
			errs = append(errs, globals.featureGate.Validate()...)
		}
	}
	return errs
}
