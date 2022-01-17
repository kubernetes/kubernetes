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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ContainerRuntimeOptions defines options for the container runtime.
type ContainerRuntimeOptions struct {
	// General Options.

	// ContainerRuntime is the container runtime to use.
	ContainerRuntime string
	// RuntimeCgroups that container runtime is expected to be isolated in.
	RuntimeCgroups string

	// Docker-specific options.

	// PodSandboxImage is the image whose network/ipc namespaces
	// containers in each pod will use.
	PodSandboxImage string
	// DockerEndpoint is the path to the docker endpoint to communicate with.
	DockerEndpoint string
	// If no pulling progress is made before the deadline imagePullProgressDeadline,
	// the image pulling will be cancelled. Defaults to 1m0s.
	// +optional
	ImagePullProgressDeadline metav1.Duration

	// Network plugin options.

	// networkPluginName is the name of the network plugin to be invoked for
	// various events in kubelet/pod lifecycle
	NetworkPluginName string
	// NetworkPluginMTU is the MTU to be passed to the network plugin,
	// and overrides the default MTU for cases where it cannot be automatically
	// computed (such as IPSEC).
	NetworkPluginMTU int32
	// CNIConfDir is the full path of the directory in which to search for
	// CNI config files
	CNIConfDir string
	// CNIBinDir is the full path of the directory in which to search for
	// CNI plugin binaries
	CNIBinDir string
	// CNICacheDir is the full path of the directory in which CNI should store
	// cache files
	CNICacheDir string

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
	dockerOnlyWarning := "This docker-specific flag only works when container-runtime is set to docker."

	// General settings.
	fs.StringVar(&s.ContainerRuntime, "container-runtime", s.ContainerRuntime, "The container runtime to use. Possible value: 'remote'.")
	fs.MarkDeprecated("container-runtime", "will be removed in 1.27 as the only valid value is 'remote'")
	fs.StringVar(&s.RuntimeCgroups, "runtime-cgroups", s.RuntimeCgroups, "Optional absolute name of cgroups to create and run the runtime in.")

	// Image credential provider settings.
	fs.StringVar(&s.ImageCredentialProviderConfigFile, "image-credential-provider-config", s.ImageCredentialProviderConfigFile, "The path to the credential provider plugin config file.")
	fs.StringVar(&s.ImageCredentialProviderBinDir, "image-credential-provider-bin-dir", s.ImageCredentialProviderBinDir, "The path to the directory where credential provider plugin binaries are located.")
}
