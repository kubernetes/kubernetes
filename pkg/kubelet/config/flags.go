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

package config

import (
	"github.com/spf13/pflag"
)

// ContainerRuntimeOptions defines options for the container runtime.
type ContainerRuntimeOptions struct {
	// General Options.

	// RuntimeCgroups that container runtime is expected to be isolated in.
	RuntimeCgroups string
	// Image credential provider plugin options

	// ImageCredentialProviderConfigPath is the path to the credential provider plugin config file or directory.
	// If a directory is specified, all .json, .yaml, or .yml files in the directory are loaded and merged
	// in lexicographical order. This config file(s) specify what credential providers are enabled
	// and invoked by the kubelet. The plugin config should contain information about what plugin binary
	// to execute and what container images the plugin should be called for.
	// +optional
	ImageCredentialProviderConfigPath string
	// ImageCredentialProviderBinDir is the path to the directory where credential provider plugin
	// binaries exist. The name of each plugin binary is expected to match the name of the plugin
	// specified in imageCredentialProviderConfigFile.
	// +optional
	ImageCredentialProviderBinDir string
}

// AddFlags adds flags to the container runtime, according to ContainerRuntimeOptions.
func (s *ContainerRuntimeOptions) AddFlags(fs *pflag.FlagSet) {
	var tmp string
	// General settings.
	fs.StringVar(&s.RuntimeCgroups, "runtime-cgroups", s.RuntimeCgroups, "Optional absolute name of cgroups to create and run the runtime in.")
	fs.StringVar(&tmp, "pod-infra-container-image", "", "Specified image will not be pruned by the image garbage collector. CRI implementations have their own configuration to set this image.")
	_ = fs.MarkDeprecated("pod-infra-container-image", "will be removed in 1.35. Image garbage collector will get sandbox image information from CRI.")

	// Image credential provider settings.
	fs.StringVar(&s.ImageCredentialProviderConfigPath, "image-credential-provider-config", s.ImageCredentialProviderConfigPath, "Path to a credential provider plugin config file (JSON/YAML/YML) or a directory of such files (merged in lexicographical order; non-recursive search).")
	fs.StringVar(&s.ImageCredentialProviderBinDir, "image-credential-provider-bin-dir", s.ImageCredentialProviderBinDir, "The path to the directory where credential provider plugin binaries are located.")
}
