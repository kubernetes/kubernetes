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
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// AdmissionOptions holds the admission options
type AdmissionOptions struct {
	PluginNames              []string
	ConfigFile               string
	Plugins                  *admission.Plugins
	genericPluginInitializer admission.PluginInitializer
}

// NewAdmissionOptions creates a new instance of AdmissionOptions
func NewAdmissionOptions(plugins *admission.Plugins) *AdmissionOptions {
	return &AdmissionOptions{
		Plugins:                  plugins,
		PluginNames:              []string{},
		genericPluginInitializer: nil,
	}
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
func (a *AdmissionOptions) ApplyTo(authz authorizer.Authorizer, restConfig *rest.Config, serverCfg *server.Config, sharedInformers informers.SharedInformerFactory, pluginInitializers ...admission.PluginInitializer) error {
	pluginsConfigProvider, err := admission.ReadAdmissionConfiguration(a.PluginNames, a.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read plugin config: %v", err)
	}

	// init generic plugin initalizer
	if a.genericPluginInitializer == nil {
		clientset, err := kubernetes.NewForConfig(restConfig)
		if err != nil {
			return err
		}
		genericInitializer, err := initializer.New(clientset, sharedInformers, authz)
		if err != nil {
			return err
		}
		a.genericPluginInitializer = genericInitializer
	}

	initializersChain := admission.PluginInitializers{}
	pluginInitializers = append(pluginInitializers, a.genericPluginInitializer)
	initializersChain = append(initializersChain, pluginInitializers...)
	admissionChain, err := a.Plugins.NewFromPlugins(a.PluginNames, pluginsConfigProvider, initializersChain)
	if err != nil {
		return err
	}

	serverCfg.AdmissionControl = admissionChain
	return nil
}
