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
	"sort"
	"strings"
	"sync"

	"github.com/spf13/pflag"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

// DefaultComponentGlobalsRegistry is the global var to store the effective versions and feature gates for all components for easy access.
// Example usage:
// // register the component effective version and feature gate first
// _, _ = utilversion.DefaultComponentGlobalsRegistry.ComponentGlobalsOrRegister(utilversion.DefaultKubeComponent, utilversion.DefaultKubeEffectiveVersion(), utilfeature.DefaultMutableFeatureGate)
// wardleEffectiveVersion := utilversion.NewEffectiveVersion("1.2")
// wardleFeatureGate := featuregate.NewFeatureGate()
// utilruntime.Must(utilversion.DefaultComponentGlobalsRegistry.Register(apiserver.WardleComponentName, wardleEffectiveVersion, wardleFeatureGate, false))
//
//	cmd := &cobra.Command{
//	 ...
//		// call DefaultComponentGlobalsRegistry.Set() in PersistentPreRunE
//		PersistentPreRunE: func(*cobra.Command, []string) error {
//			if err := utilversion.DefaultComponentGlobalsRegistry.Set(); err != nil {
//				return err
//			}
//	 ...
//		},
//		RunE: func(c *cobra.Command, args []string) error {
//			// call utilversion.DefaultComponentGlobalsRegistry.Validate() somewhere
//		},
//	}
//
// flags := cmd.Flags()
// // add flags
// utilversion.DefaultComponentGlobalsRegistry.AddFlags(flags)
var DefaultComponentGlobalsRegistry ComponentGlobalsRegistry = NewComponentGlobalsRegistry()

const (
	DefaultKubeComponent = "kube"
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
	// AddFlags adds flags of "--emulated-version" and "--feature-gates"
	AddFlags(fs *pflag.FlagSet)
	// Set sets the flags for all global variables for all components registered.
	Set() error
	// SetAllComponents calls the Validate() function for all the global variables for all components registered.
	Validate() []error
}

type componentGlobalsRegistry struct {
	componentGlobals map[string]ComponentGlobals
	mutex            sync.RWMutex
	// map of component name to emulation version set from the flag.
	emulationVersionConfig cliflag.ConfigurationMap
	// map of component name to the list of feature gates set from the flag.
	featureGatesConfig map[string][]string
}

func NewComponentGlobalsRegistry() ComponentGlobalsRegistry {
	return &componentGlobalsRegistry{
		componentGlobals: make(map[string]ComponentGlobals),
	}
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
		if err := featureGate.SetEmulationVersion(effectiveVersion.EmulationVersion()); err != nil {
			return err
		}
	}
	c := ComponentGlobals{effectiveVersion: effectiveVersion, featureGate: featureGate}
	r.componentGlobals[component] = c
	return nil
}

func (r *componentGlobalsRegistry) Register(component string, effectiveVersion MutableEffectiveVersion, featureGate featuregate.MutableVersionedFeatureGate, override bool) error {
	if effectiveVersion == nil {
		return fmt.Errorf("cannot register nil effectiveVersion")
	}
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

func (r *componentGlobalsRegistry) knownFeatures() []string {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	var known []string
	for component, globals := range r.componentGlobals {
		if globals.featureGate == nil {
			continue
		}
		for _, f := range globals.featureGate.KnownFeatures() {
			known = append(known, component+":"+f)
		}
	}
	sort.Strings(known)
	return known
}

func (r *componentGlobalsRegistry) versionFlagOptions(isEmulation bool) []string {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	var vs []string
	for component, globals := range r.componentGlobals {
		binaryVer := globals.effectiveVersion.BinaryVersion()
		if isEmulation {
			// emulated version could be between binaryMajor.{binaryMinor} and binaryMajor.{binaryMinor}
			// TODO: change to binaryMajor.{binaryMinor-1} and binaryMajor.{binaryMinor} in 1.32
			vs = append(vs, fmt.Sprintf("%s=%s..%s (default=%s)", component,
				binaryVer.SubtractMinor(0).String(), binaryVer.String(), globals.effectiveVersion.EmulationVersion().String()))
		} else {
			// min compatibility version could be between binaryMajor.{binaryMinor-1} and binaryMajor.{binaryMinor}
			vs = append(vs, fmt.Sprintf("%s=%s..%s (default=%s)", component,
				binaryVer.SubtractMinor(1).String(), binaryVer.String(), globals.effectiveVersion.MinCompatibilityVersion().String()))
		}
	}
	sort.Strings(vs)
	return vs
}

func (r *componentGlobalsRegistry) AddFlags(fs *pflag.FlagSet) {
	if r == nil {
		return
	}
	r.mutex.Lock()
	for _, globals := range r.componentGlobals {
		if globals.featureGate != nil {
			globals.featureGate.Close()
		}
	}
	r.emulationVersionConfig = make(cliflag.ConfigurationMap)
	r.featureGatesConfig = make(map[string][]string)
	r.mutex.Unlock()

	fs.Var(&r.emulationVersionConfig, "emulated-version", ""+
		"The versions different components emulate their capabilities (APIs, features, ...) of.\n"+
		"If set, the component will emulate the behavior of this version instead of the underlying binary version.\n"+
		"Version format could only be major.minor, for example: '--emulated-version=wardle=1.2,kube=1.31'. Options are:\n"+strings.Join(r.versionFlagOptions(true), "\n"))

	fs.Var(cliflag.NewColonSeparatedMultimapStringStringAllowDefaultEmptyKey(&r.featureGatesConfig), "feature-gates", "Comma-separated list of component:key=value pairs that describe feature gates for alpha/experimental features of different components.\n"+
		"If the component is not specified, defaults to \"kube\". This flag can be repeatedly invoked. For example: --feature-gates 'wardle:featureA=true,wardle:featureB=false' --feature-gates 'kube:featureC=true'"+
		"Options are:\n"+strings.Join(r.knownFeatures(), "\n"))
}

func (r *componentGlobalsRegistry) Set() error {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	for comp, emuVer := range r.emulationVersionConfig {
		if _, ok := r.componentGlobals[comp]; !ok {
			return fmt.Errorf("component not registered: %s", comp)
		}
		klog.V(2).Infof("setting %s:emulation version to %s\n", comp, emuVer)
		v, err := version.Parse(emuVer)
		if err != nil {
			return err
		}
		r.componentGlobals[comp].effectiveVersion.SetEmulationVersion(v)
	}
	// Set feature gate emulation version before setting feature gate flag values.
	for comp, globals := range r.componentGlobals {
		if globals.featureGate == nil {
			continue
		}
		klog.V(2).Infof("setting %s:feature gate emulation version to %s\n", comp, globals.effectiveVersion.EmulationVersion().String())
		if err := globals.featureGate.SetEmulationVersion(globals.effectiveVersion.EmulationVersion()); err != nil {
			return err
		}
	}
	for comp, fg := range r.featureGatesConfig {
		if comp == "" {
			comp = DefaultKubeComponent
		}
		if _, ok := r.componentGlobals[comp]; !ok {
			return fmt.Errorf("component not registered: %s", comp)
		}
		featureGate := r.componentGlobals[comp].featureGate
		if featureGate == nil {
			return fmt.Errorf("component featureGate not registered: %s", comp)
		}
		flagVal := strings.Join(fg, ",")
		klog.V(2).Infof("setting %s:feature-gates=%s\n", comp, flagVal)
		if err := featureGate.Set(flagVal); err != nil {
			return err
		}
	}
	return nil
}

func (r *componentGlobalsRegistry) Validate() []error {
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
