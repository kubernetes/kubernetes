/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package app

// This file exists to force the desired plugin implementations to be linked.
import (
	// Credential providers
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider/gcp"
	// Network plugins
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network/exec"
	// Volume plugins
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/aws_ebs"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/empty_dir"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/git_repo"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/glusterfs"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/host_path"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/iscsi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/nfs"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/persistent_claim"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/rbd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/secret"
	//Cloud providers
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/aws"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/mesos"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/openstack"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/ovirt"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/rackspace"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/vagrant"
)

// ProbeVolumePlugins collects all volume plugins into an easy to use list.
func ProbeVolumePlugins() []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}

	// The list of plugins to probe is decided by the kubelet binary, not
	// by dynamic linking or other "magic".  Plugins will be analyzed and
	// initialized later.
	allPlugins = append(allPlugins, aws_ebs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, empty_dir.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, git_repo.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, secret.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, iscsi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, glusterfs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, persistent_claim.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, rbd.ProbeVolumePlugins()...)

	return allPlugins
}

// ProbeNetworkPlugins collects all compiled-in plugins
func ProbeNetworkPlugins() []network.NetworkPlugin {
	allPlugins := []network.NetworkPlugin{}

	// for each existing plugin, add to the list
	allPlugins = append(allPlugins, exec.ProbeNetworkPlugins()...)

	return allPlugins
}
