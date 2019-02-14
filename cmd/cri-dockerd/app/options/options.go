/*
Copyright 2015 The Kubernetes Authors.

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

// Package options contains all of the primary arguments for cri-dockerd.
package options

import (
	"runtime"

	"k8s.io/kubernetes/cmd/cri-dockerd/app/config"

	"github.com/spf13/pflag"
)

// DockerCRIFlags contains configuration flags for the cri-dockerd.
type DockerCRIFlags struct {
	// Container-runtime-specific options.
	config.ContainerRuntimeOptions
	// remoteRuntimeEndpoint is the endpoint of remote runtime service
	RemoteRuntimeEndpoint string
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string
}

// NewDockerCRIFlags will create a new DockerCRIFlags with default values
func NewDockerCRIFlags() *DockerCRIFlags {
	remoteRuntimeEndpoint := ""
	if runtime.GOOS == "linux" {
		remoteRuntimeEndpoint = "unix:///var/run/dockershim.sock"
	} else if runtime.GOOS == "windows" {
		remoteRuntimeEndpoint = "npipe:////./pipe/dockershim"
	}

	return &DockerCRIFlags{
		ContainerRuntimeOptions: *NewContainerRuntimeOptions(),
		NonMasqueradeCIDR:       "10.0.0.0/8",
		RemoteRuntimeEndpoint:   remoteRuntimeEndpoint,
	}
}

// DockerCRIServer encapsulates all of the parameters necessary for starting up
// a kubelet. These can either be set via command line or directly.
type DockerCRIServer struct {
	DockerCRIFlags
}

// AddFlags adds flags for a specific DockerCRIServer to the specified FlagSet
func (s *DockerCRIServer) AddFlags(fs *pflag.FlagSet) {
	s.DockerCRIFlags.AddFlags(fs)
}

// AddFlags adds flags for a specific DockerCRIFlags to the specified FlagSet
func (f *DockerCRIFlags) AddFlags(mainfs *pflag.FlagSet) {
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	defer func() {
		// Unhide deprecated flags. We want deprecated flags to show in cri-dockerd's help.
		// We have some hidden flags, but we might as well unhide these when they are deprecated,
		// as silently deprecating and removing (even hidden) things is unkind to people who use them.
		fs.VisitAll(func(f *pflag.Flag) {
			if len(f.Deprecated) > 0 {
				f.Hidden = false
			}
		})
		mainfs.AddFlagSet(fs)
	}()

	f.ContainerRuntimeOptions.AddFlags(fs)
	fs.StringVar(&f.RemoteRuntimeEndpoint, "container-runtime-endpoint", f.RemoteRuntimeEndpoint, "The endpoint of remote runtime service. Currently unix socket and tcp endpoints are supported on Linux, while npipe and tcp endpoints are supported on windows.  Examples:'unix:///var/run/dockershim.sock', 'npipe:////./pipe/dockershim'")
}
