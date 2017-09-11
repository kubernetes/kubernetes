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
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
)

// AdmissionOptions holds the admission options
type AdmissionOptions struct {
	PluginNames []string
	ConfigFile  string
	Plugins     *admission.Plugins
}

// NewAdmissionOptions creates a new instance of AdmissionOptions
// Note:
// In addition it calls RegisterAllAdmissionPlugins to register
// all generic admission plugins.
func NewAdmissionOptions() *AdmissionOptions {
	options := &AdmissionOptions{
		Plugins:     &admission.Plugins{},
		PluginNames: []string{},
	}
	server.RegisterAllAdmissionPlugins(options.Plugins)
	return options
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (a *AdmissionOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVar(&a.PluginNames, "admission-control", a.PluginNames, ""+
		"Ordered list of plug-ins to do admission control of resources into cluster. "+
		"Comma-delimited list of: "+strings.Join(a.Plugins.Registered(), ", ")+".")

	fs.StringVar(&a.ConfigFile, "admission-control-config-file", a.ConfigFile,
		"File with admission control configuration.")
}

// ApplyTo adds the admission chain to the server configuration
// the method lazily initializes a generic plugin that is appended to the list of pluginInitializers
// note this method uses:
//  genericconfig.LoopbackClientConfig
//  genericconfig.SharedInformerFactory
//  genericconfig.Authorizer
func (a *AdmissionOptions) ApplyTo(c *server.Config, informers informers.SharedInformerFactory, pluginInitializers ...admission.PluginInitializer) error {
	pluginsConfigProvider, err := admission.ReadAdmissionConfiguration(a.PluginNames, a.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read plugin config: %v", err)
	}

	clientset, err := kubernetes.NewForConfig(c.LoopbackClientConfig)
	if err != nil {
		return err
	}
	genericInitializer, err := initializer.New(clientset, informers, c.Authorizer)
	if err != nil {
		return err
	}
	initializersChain := admission.PluginInitializers{}
	pluginInitializers = append(pluginInitializers, genericInitializer)
	initializersChain = append(initializersChain, pluginInitializers...)

	admissionChain, err := a.Plugins.NewFromPlugins(a.PluginNames, pluginsConfigProvider, initializersChain)
	if err != nil {
		return err
	}

	c.AdmissionControl = admissionChain
	return nil
}

func (a *AdmissionOptions) Validate() []error {
	errs := []error{}
	return errs
}
