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
	"fmt"

	"github.com/spf13/pflag"
)

// ContainerRuntimeOptions defines options for the container runtime.
type ContainerRuntimeOptions struct {
	// General Options.

	// RuntimeCgroups that container runtime is expected to be isolated in.
	RuntimeCgroups string
	// PodSandboxImage is the image whose network/ipc namespaces
	// containers in each pod will use.
	PodSandboxImage string
	// Image credential provider plugin options

	// ImageCredentialProviderConfigFile is the path to the credential provider plugin config file.
	// This config file is a specification for what credential providers are enabled and invoked
	// by the kubelet. The plugin config should contain information about what plugin binary
	// to execute and what container images the plugin should be called for.
	// +optional
	ImageCredentialProviderConfigFile string
	// ImageCredentialProviderBinDir is the path to the directory where credential provider plugin
	// binaries exist. The name of each plugin binary is expected to match the name of the plugin
	// specified in imageCredentialProviderConfigFile.
	// +optional
	ImageCredentialProviderBinDir string
}

// AddFlags adds flags to the container runtime, according to ContainerRuntimeOptions.
func (s *ContainerRuntimeOptions) AddFlags(fs *pflag.FlagSet) {
	// General settings.
	fs.StringVar(&s.RuntimeCgroups, "runtime-cgroups", s.RuntimeCgroups, "Optional absolute name of cgroups to create and run the runtime in.")
	fs.StringVar(&s.PodSandboxImage, "pod-infra-container-image", s.PodSandboxImage, fmt.Sprintf("Specified image will not be pruned by the image garbage collector. CRI implementations have their own configuration to set this image."))
	fs.MarkDeprecated("pod-infra-container-image", "will be removed in a future release. Image garbage collector will get sandbox image information from CRI.")

	// Image credential provider settings.
	fs.StringVar(&s.ImageCredentialProviderConfigFile, "image-credential-provider-config", s.ImageCredentialProviderConfigFile, "The path to the credential provider plugin config file.")
	fs.StringVar(&s.ImageCredentialProviderBinDir, "image-credential-provider-bin-dir", s.ImageCredentialProviderBinDir, "The path to the directory where credential provider plugin binaries are located.")
}
