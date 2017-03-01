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
	"runtime"

	"github.com/spf13/pflag"
)

const (
	// When these values are updated, also update test/e2e/framework/util.go
	defaultPodSandboxImageName    = "gcr.io/google_containers/pause"
	defaultPodSandboxImageVersion = "3.0"
	defaultPodSandboxImage        = defaultPodSandboxImageName +
		"-" + runtime.GOARCH + ":" +
		defaultPodSandboxImageVersion
)

type DockerOptions struct {
	// PodSandboxImage is the image whose network/ipc namespaces
	// containers in each pod will use.
	PodSandboxImage string
	// DockerEndpoint is the path to the docker endpoint to communicate with.
	DockerEndpoint string
	// networkPluginName is the name of the network plugin to be invoked for
	// various events in kubelet/pod lifecycle
	NetworkPluginName string
	// NetworkPluginMTU is the MTU to be passed to the network plugin,
	// and overrides the default MTU for cases where it cannot be automatically
	// computed (such as IPSEC).
	NetworkPluginMTU int32
	// NetworkPluginDir is the full path of the directory in which to search
	// for network plugins (and, for backwards-compat, CNI config files)
	NetworkPluginDir string
	// CNIConfDir is the full path of the directory in which to search for
	// CNI config files
	CNIConfDir string
	// CNIBinDir is the full path of the directory in which to search for
	// CNI plugin binaries
	CNIBinDir string
	// ExecHandlerName is the handler to use when executing a command
	// in a container. Valid values are 'native' and 'nsenter'. Defaults to
	// 'native'.
	ExecHandlerName string
}

// NewKubeletServer will create a new KubeletServer with default values.
func NewDockerOptions() *DockerOptions {
	return &DockerOptions{
		ExecHandlerName: "native",
		PodSandboxImage: defaultPodSandboxImage,
	}
}

func (s *DockerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.PodSandboxImage, "pod-infra-container-image", s.PodSandboxImage, "The image whose network/ipc namespaces containers in each pod will use.")
	fs.StringVar(&s.DockerEndpoint, "docker-endpoint", s.DockerEndpoint, "Use this for the docker endpoint to communicate with")
	// TODO(#40229): Remove the docker-exec-handler flag.
	fs.StringVar(&s.ExecHandlerName, "docker-exec-handler", s.ExecHandlerName, "Handler to use when executing a command in a container. Valid values are 'native' and 'nsenter'. Defaults to 'native'.")
	fs.MarkDeprecated("docker-exec-handler", "this flag will be removed and only the 'native' handler will be supported in the future.")
	fs.StringVar(&s.NetworkPluginName, "network-plugin", s.NetworkPluginName, "<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle")
	fs.StringVar(&s.NetworkPluginDir, "network-plugin-dir", s.NetworkPluginDir, "<Warning: Alpha feature> The full path of the directory in which to search for network plugins or CNI config")
	fs.StringVar(&s.CNIConfDir, "cni-conf-dir", s.CNIConfDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI config files. Default: /etc/cni/net.d")
	fs.StringVar(&s.CNIBinDir, "cni-bin-dir", s.CNIBinDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI plugin binaries. Default: /opt/cni/bin")
	fs.Int32Var(&s.NetworkPluginMTU, "network-plugin-mtu", s.NetworkPluginMTU, "<Warning: Alpha feature> The MTU to be passed to the network plugin, to override the default. Set to 0 to use the default 1460 MTU.")
}
