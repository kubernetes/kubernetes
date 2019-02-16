/*
Copyright 2018 The Kubernetes Authors.

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
	"github.com/spf13/pflag"

	// libs that provide registration functions
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/kubernetes/pkg/version/verflag"

	// ensure libs have a chance to globally register their flags
	_ "k8s.io/kubernetes/pkg/credentialprovider/azure"
	_ "k8s.io/kubernetes/pkg/credentialprovider/gcp"
)

// AddCustomGlobalFlags explicitly registers flags that libraries (glog, verflag, etc.) register
// against the global flagsets from "flag" and "github.com/spf13/pflag".
// We do this in order to prevent unwanted flags from leaking into the Kubelet's flagset.
func AddCustomGlobalFlags(fs *pflag.FlagSet) {
	addCadvisorFlags(fs)
	addCredentialProviderFlags(fs)
	verflag.AddFlags(fs)
}

// addCredentialProviderFlags adds flags from k8s.io/kubernetes/pkg/credentialprovider.
func addCredentialProviderFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with pflag.CommandLine

	// TODO(#58034): This is not a static file, so it's not quite as straightforward as --google-json-key.
	// We need to figure out how ACR users can dynamically provide pull credentials before we can deprecate this.
	globalflag.RegisterPflag(fs, "azure-container-registry-config")
}
