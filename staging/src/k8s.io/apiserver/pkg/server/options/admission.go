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

package options

import (
	"fmt"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/initialization"
	"k8s.io/apiserver/pkg/admission/plugin/namespace/lifecycle"
	mutatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating"
	validatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/validating"
	apiserverapi "k8s.io/apiserver/pkg/apis/apiserver"
	apiserverapiv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

var configScheme = runtime.NewScheme()

func init() {
	utilruntime.Must(apiserverapi.AddToScheme(configScheme))
	utilruntime.Must(apiserverapiv1alpha1.AddToScheme(configScheme))
}

// AdmissionOptions holds the admission options
type AdmissionOptions struct {
	// RecommendedPluginOrder holds an ordered list of plugin names we recommend to use by default
	RecommendedPluginOrder []string
	// DefaultOffPlugins is a set of plugin names that is disabled by default
	DefaultOffPlugins sets.String

	// EnablePlugins indicates plugins to be enabled passed through `--enable-admission-plugins`.
	EnablePlugins []string
	// DisablePlugins indicates plugins to be disabled passed through `--disable-admission-plugins`.
	DisablePlugins []string
	// ConfigFile is the file path with admission control configuration.
	ConfigFile string
	// Plugins contains all registered plugins.
	Plugins *admission.Plugins
}

// NewAdmissionOptions creates a new instance of AdmissionOptions
// Note:
//  In addition it calls RegisterAllAdmissionPlugins to register
//  all generic admission plugins.
//
//  Provides the list of RecommendedPluginOrder that holds sane values
//  that can be used by servers that don't care about admission chain.
//  Servers that do care can overwrite/append that field after creation.
func NewAdmissionOptions() *AdmissionOptions {
	options := &AdmissionOptions{
		Plugins: admission.NewPlugins(),
		// This list is mix of mutating admission plugins and validating
		// admission plugins. The apiserver always runs the validating ones
		// after all the mutating ones, so their relative order in this list
		// doesn't matter.
		RecommendedPluginOrder: []string{lifecycle.PluginName, initialization.PluginName, mutatingwebhook.PluginName, validatingwebhook.PluginName},
		DefaultOffPlugins:      sets.NewString(initialization.PluginName),
	}
	server.RegisterAllAdmissionPlugins(options.Plugins)
	return options
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (a *AdmissionOptions) AddFlags(fs *pflag.FlagSet) {
	if a == nil {
		return
	}

	fs.StringSliceVar(&a.EnablePlugins, "enable-admission-plugins", a.EnablePlugins, ""+
		"admission plugins that should be enabled in addition to default enabled ones ("+
		strings.Join(a.defaultEnabledPluginNames(), ", ")+"). "+
		"Comma-delimited list of admission plugins: "+strings.Join(a.Plugins.Registered(), ", ")+". "+
		"The order of plugins in this flag does not matter.")
	fs.StringSliceVar(&a.DisablePlugins, "disable-admission-plugins", a.DisablePlugins, ""+
		"admission plugins that should be disabled although they are in the default enabled plugins list ("+
		strings.Join(a.defaultEnabledPluginNames(), ", ")+"). "+
		"Comma-delimited list of admission plugins: "+strings.Join(a.Plugins.Registered(), ", ")+". "+
		"The order of plugins in this flag does not matter.")
	fs.StringVar(&a.ConfigFile, "admission-control-config-file", a.ConfigFile,
		"File with admission control configuration.")
}

// ApplyTo adds the admission chain to the server configuration.
// In case admission plugin names were not provided by a custer-admin they will be prepared from the recommended/default values.
// In addition the method lazily initializes a generic plugin that is appended to the list of pluginInitializers
// note this method uses:
//  genericconfig.Authorizer
func (a *AdmissionOptions) ApplyTo(
	c *server.Config,
	informers informers.SharedInformerFactory,
	kubeAPIServerClientConfig *rest.Config,
	scheme *runtime.Scheme,
	pluginInitializers ...admission.PluginInitializer,
) error {
	if a == nil {
		return nil
	}

	// Admission need scheme to construct admission initializer.
	if scheme == nil {
		return fmt.Errorf("admission depends on a scheme, it cannot be nil")
	}

	// Admission depends on CoreAPI to set SharedInformerFactory and ClientConfig.
	if informers == nil {
		return fmt.Errorf("admission depends on a Kubernetes core API shared informer, it cannot be nil")
	}

	pluginNames := a.enabledPluginNames()

	pluginsConfigProvider, err := admission.ReadAdmissionConfiguration(pluginNames, a.ConfigFile, configScheme)
	if err != nil {
		return fmt.Errorf("failed to read plugin config: %v", err)
	}

	clientset, err := kubernetes.NewForConfig(kubeAPIServerClientConfig)
	if err != nil {
		return err
	}
	genericInitializer := initializer.New(clientset, informers, c.Authorization.Authorizer, scheme)
	initializersChain := admission.PluginInitializers{}
	pluginInitializers = append(pluginInitializers, genericInitializer)
	initializersChain = append(initializersChain, pluginInitializers...)

	admissionChain, err := a.Plugins.NewFromPlugins(pluginNames, pluginsConfigProvider, initializersChain, admission.DecoratorFunc(admissionmetrics.WithControllerMetrics))
	if err != nil {
		return err
	}

	c.AdmissionControl = admissionmetrics.WithStepMetrics(admissionChain)
	return nil
}

// Validate verifies flags passed to AdmissionOptions.
func (a *AdmissionOptions) Validate() []error {
	if a == nil {
		return nil
	}

	errs := []error{}

	registeredPlugins := sets.NewString(a.Plugins.Registered()...)
	for _, name := range a.EnablePlugins {
		if !registeredPlugins.Has(name) {
			errs = append(errs, fmt.Errorf("enable-admission-plugins plugin %q is unknown", name))
		}
	}

	for _, name := range a.DisablePlugins {
		if !registeredPlugins.Has(name) {
			errs = append(errs, fmt.Errorf("disable-admission-plugins plugin %q is unknown", name))
		}
	}

	enablePlugins := sets.NewString(a.EnablePlugins...)
	disablePlugins := sets.NewString(a.DisablePlugins...)
	if len(enablePlugins.Intersection(disablePlugins).List()) > 0 {
		errs = append(errs, fmt.Errorf("%v in enable-admission-plugins and disable-admission-plugins "+
			"overlapped", enablePlugins.Intersection(disablePlugins).List()))
	}

	// Verify RecommendedPluginOrder.
	recommendPlugins := sets.NewString(a.RecommendedPluginOrder...)
	intersections := registeredPlugins.Intersection(recommendPlugins)
	if !intersections.Equal(recommendPlugins) {
		// Developer error, this should never run in.
		errs = append(errs, fmt.Errorf("plugins %v in RecommendedPluginOrder are not registered",
			recommendPlugins.Difference(intersections).List()))
	}
	if !intersections.Equal(registeredPlugins) {
		// Developer error, this should never run in.
		errs = append(errs, fmt.Errorf("plugins %v registered are not in RecommendedPluginOrder",
			registeredPlugins.Difference(intersections).List()))
	}

	return errs
}

// enabledPluginNames makes use of RecommendedPluginOrder, DefaultOffPlugins,
// EnablePlugins, DisablePlugins fields
// to prepare a list of ordered plugin names that are enabled.
func (a *AdmissionOptions) enabledPluginNames() []string {
	allOffPlugins := append(a.DefaultOffPlugins.List(), a.DisablePlugins...)
	disabledPlugins := sets.NewString(allOffPlugins...)
	enabledPlugins := sets.NewString(a.EnablePlugins...)
	disabledPlugins = disabledPlugins.Difference(enabledPlugins)

	orderedPlugins := []string{}
	for _, plugin := range a.RecommendedPluginOrder {
		if !disabledPlugins.Has(plugin) {
			orderedPlugins = append(orderedPlugins, plugin)
		}
	}

	return orderedPlugins
}

//Return names of plugins which are enabled by default
func (a *AdmissionOptions) defaultEnabledPluginNames() []string {
	defaultOnPluginNames := []string{}
	for _, pluginName := range a.RecommendedPluginOrder {
		if !a.DefaultOffPlugins.Has(pluginName) {
			defaultOnPluginNames = append(defaultOnPluginNames, pluginName)
		}
	}

	return defaultOnPluginNames
}
