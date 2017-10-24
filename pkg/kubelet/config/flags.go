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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ContainerRuntimeOptions struct {
	// Docker-specific options.

	// DockershimRootDirectory is the path to the dockershim root directory. Defaults to
	// /var/lib/dockershim if unset. Exposed for integration testing (e.g. in OpenShift).
	DockershimRootDirectory string
	// Enable dockershim only mode.
	ExperimentalDockershim bool
	// This flag, if set, disables use of a shared PID namespace for pods running in the docker CRI runtime.
	// A shared PID namespace is the only option in non-docker runtimes and is required by the CRI. The ability to
	// disable it for docker will be removed unless a compelling use case is discovered with widespread use.
	// TODO: Remove once we no longer support disabling shared PID namespace (https://issues.k8s.io/41938)
	DockerDisableSharedPID bool
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

	// rkt-specific options.

	// rktPath is the path of rkt binary. Leave empty to use the first rkt in $PATH.
	RktPath string
	// rktApiEndpoint is the endpoint of the rkt API service to communicate with.
	RktAPIEndpoint string
	// rktStage1Image is the image to use as stage1. Local paths and
	// http/https URLs are supported.
	RktStage1Image string
}

func (s *ContainerRuntimeOptions) AddFlags(fs *pflag.FlagSet) {
	// Docker-specific settings.
	fs.BoolVar(&s.ExperimentalDockershim, "experimental-dockershim", s.ExperimentalDockershim, "Enable dockershim only mode. In this mode, kubelet will only start dockershim without any other functionalities. This flag only serves test purpose, please do not use it unless you are conscious of what you are doing. [default=false]")
	fs.MarkHidden("experimental-dockershim")
	fs.StringVar(&s.DockershimRootDirectory, "experimental-dockershim-root-directory", s.DockershimRootDirectory, "Path to the dockershim root directory.")
	fs.MarkHidden("experimental-dockershim-root-directory")
	fs.BoolVar(&s.DockerDisableSharedPID, "docker-disable-shared-pid", s.DockerDisableSharedPID, "The Container Runtime Interface (CRI) defaults to using a shared PID namespace for containers in a pod when running with Docker 1.13.1 or higher. Setting this flag reverts to the previous behavior of isolated PID namespaces. This ability will be removed in a future Kubernetes release.")
	fs.StringVar(&s.PodSandboxImage, "pod-infra-container-image", s.PodSandboxImage, "The image whose network/ipc namespaces containers in each pod will use.")
	fs.StringVar(&s.DockerEndpoint, "docker-endpoint", s.DockerEndpoint, "Use this for the docker endpoint to communicate with")
	fs.DurationVar(&s.ImagePullProgressDeadline.Duration, "image-pull-progress-deadline", s.ImagePullProgressDeadline.Duration, "If no pulling progress is made before this deadline, the image pulling will be cancelled.")

	// Network plugin settings. Shared by both docker and rkt.
	fs.StringVar(&s.NetworkPluginName, "network-plugin", s.NetworkPluginName, "<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle")
	fs.StringVar(&s.CNIConfDir, "cni-conf-dir", s.CNIConfDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI config files. Default: /etc/cni/net.d")
	fs.StringVar(&s.CNIBinDir, "cni-bin-dir", s.CNIBinDir, "<Warning: Alpha feature> The full path of the directory in which to search for CNI plugin binaries. Default: /opt/cni/bin")
	fs.Int32Var(&s.NetworkPluginMTU, "network-plugin-mtu", s.NetworkPluginMTU, "<Warning: Alpha feature> The MTU to be passed to the network plugin, to override the default. Set to 0 to use the default 1460 MTU.")

	// Rkt-specific settings.
	fs.StringVar(&s.RktPath, "rkt-path", s.RktPath, "Path of rkt binary. Leave empty to use the first rkt in $PATH.  Only used if --container-runtime='rkt'.")
	fs.StringVar(&s.RktAPIEndpoint, "rkt-api-endpoint", s.RktAPIEndpoint, "The endpoint of the rkt API service to communicate with. Only used if --container-runtime='rkt'.")
	fs.StringVar(&s.RktStage1Image, "rkt-stage1-image", s.RktStage1Image, "image to use as stage1. Local paths and http/https URLs are supported. If empty, the 'stage1.aci' in the same directory as '--rkt-path' will be used.")
	fs.MarkDeprecated("rkt-stage1-image", "Will be removed in a future version. The default stage1 image will be specified by the rkt configurations, see https://github.com/coreos/rkt/blob/master/Documentation/configuration.md for more details.")

}
