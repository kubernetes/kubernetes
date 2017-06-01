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
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/admission"
)

// AdmissionOptions holds the admission options
type AdmissionOptions struct {
	Control           string
	ControlConfigFile string
	Plugins           *admission.Plugins
}

// NewAdmissionOptions creates a new instance of AdmissionOptions
func NewAdmissionOptions(plugins *admission.Plugins) *AdmissionOptions {
	return &AdmissionOptions{
		Plugins: plugins,
		Control: "AlwaysAdmit",
	}
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (a *AdmissionOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&a.Control, "admission-control", a.Control, ""+
		"Ordered list of plug-ins to do admission control of resources into cluster. "+
		"Comma-delimited list of: "+strings.Join(a.Plugins.Registered(), ", ")+".")

	fs.StringVar(&a.ControlConfigFile, "admission-control-config-file", a.ControlConfigFile,
		"File with admission control configuration.")
}
