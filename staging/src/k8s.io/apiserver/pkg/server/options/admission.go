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
	PluginsNames             []string
	ConfigFile               string
	Plugins                  *admission.Plugins
	genericPluginInitializer admission.PluginInitializer
}

// NewAdmissionOptions creates a new instance of AdmissionOptions
func NewAdmissionOptions(plugins *admission.Plugins) *AdmissionOptions {
	return &AdmissionOptions{
		Plugins:                  plugins,
		PluginsNames:             []string{},
		genericPluginInitializer: nil,
	}
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (a *AdmissionOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVar(&a.PluginsNames, "admission-control", a.PluginsNames, ""+
		"Ordered list of plug-ins to do admission control of resources into cluster. "+
		"Comma-delimited list of: "+strings.Join(a.Plugins.Registered(), ", ")+".")

	fs.StringVar(&a.ConfigFile, "admission-control-config-file", a.ConfigFile,
		"File with admission control configuration.")
}

// ApplyTo adds the admission chain to the server configuration
// note that pluginIntializer is optional, a generic plugin intializer will always be provided and appended
// to the list of plugin initializers.
func (a *AdmissionOptions) ApplyTo(pluginInitializer admission.PluginInitializer, authz authorizer.Authorizer, restConfig *rest.Config, serverCfg *server.Config, sharedInformers informers.SharedInformerFactory) error {
	pluginsConfigProvider, err := admission.ReadAdmissionConfiguration(a.PluginsNames, a.ConfigFile)
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

	pluginInitializers := admission.PluginInitializers{a.genericPluginInitializer}
	if pluginInitializer != nil {
		pluginInitializers = append(pluginInitializers, pluginInitializer)
	}
	admissionChain, err := a.Plugins.NewFromPlugins(a.PluginsNames, pluginsConfigProvider, pluginInitializers)
	if err != nil {
		return err
	}

	serverCfg.AdmissionControl = admissionChain
	return nil
}
