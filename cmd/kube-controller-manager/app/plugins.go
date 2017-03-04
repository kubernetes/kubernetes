/*
Copyright 2014 The Kubernetes Authors.

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

import (
	// This file exists to force the desired plugin implementations to be linked.
	// This should probably be part of some configuration fed into the build for a
	// given binary target.

	"fmt"

	// Cloud providers
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"

	// Volume plugins
	"github.com/golang/glog"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/photon"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/aws_ebs"
	"k8s.io/kubernetes/pkg/volume/azure_dd"
	"k8s.io/kubernetes/pkg/volume/cinder"
	"k8s.io/kubernetes/pkg/volume/flexvolume"
	"k8s.io/kubernetes/pkg/volume/flocker"
	"k8s.io/kubernetes/pkg/volume/gce_pd"
	"k8s.io/kubernetes/pkg/volume/glusterfs"
	"k8s.io/kubernetes/pkg/volume/host_path"
	"k8s.io/kubernetes/pkg/volume/nfs"
	"k8s.io/kubernetes/pkg/volume/photon_pd"
	"k8s.io/kubernetes/pkg/volume/portworx"
	"k8s.io/kubernetes/pkg/volume/quobyte"
	"k8s.io/kubernetes/pkg/volume/rbd"
	"k8s.io/kubernetes/pkg/volume/scaleio"
	"k8s.io/kubernetes/pkg/volume/vsphere_volume"
)

// ProbeAttachableVolumePlugins collects all volume plugins for the attach/
// detach controller. VolumeConfiguration is used ot get FlexVolumePluginDir
// which specifies the directory to search for additional third party volume
// plugins.
// The list of plugins is manually compiled. This code and the plugin
// initialization code for kubelet really, really need a through refactor.
func ProbeAttachableVolumePlugins(config componentconfig.VolumeConfiguration) []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}

	allPlugins = append(allPlugins, aws_ebs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, cinder.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, flexvolume.ProbeVolumePlugins(config.FlexVolumePluginDir)...)
	allPlugins = append(allPlugins, portworx.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, vsphere_volume.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, azure_dd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, photon_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, scaleio.ProbeVolumePlugins()...)
	return allPlugins
}

// ProbeControllerVolumePlugins collects all persistent volume plugins into an
// easy to use list. Only volume plugins that implement any of
// provisioner/recycler/deleter interface should be returned.
func ProbeControllerVolumePlugins(cloud cloudprovider.Interface, config componentconfig.VolumeConfiguration) []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}

	// The list of plugins to probe is decided by this binary, not
	// by dynamic linking or other "magic".  Plugins will be analyzed and
	// initialized later.

	// Each plugin can make use of VolumeConfig.  The single arg to this func contains *all* enumerated
	// options meant to configure volume plugins.  From that single config, create an instance of volume.VolumeConfig
	// for a specific plugin and pass that instance to the plugin's ProbeVolumePlugins(config) func.

	// HostPath recycling is for testing and development purposes only!
	hostPathConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   int(config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath),
		RecyclerTimeoutIncrement: int(config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath),
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
		ProvisioningEnabled:      config.EnableHostPathProvisioning,
	}
	if err := AttemptToLoadRecycler(config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, &hostPathConfig); err != nil {
		glog.Fatalf("Could not create hostpath recycler pod from file %s: %+v", config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, err)
	}
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins(hostPathConfig)...)

	nfsConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   int(config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS),
		RecyclerTimeoutIncrement: int(config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS),
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	if err := AttemptToLoadRecycler(config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, &nfsConfig); err != nil {
		glog.Fatalf("Could not create NFS recycler pod from file %s: %+v", config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, err)
	}
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins(nfsConfig)...)
	allPlugins = append(allPlugins, glusterfs.ProbeVolumePlugins()...)
	// add rbd provisioner
	allPlugins = append(allPlugins, rbd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, quobyte.ProbeVolumePlugins()...)

	allPlugins = append(allPlugins, flocker.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, portworx.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, scaleio.ProbeVolumePlugins()...)

	if cloud != nil {
		switch {
		case aws.ProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, aws_ebs.ProbeVolumePlugins()...)
		case gce.ProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
		case openstack.ProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, cinder.ProbeVolumePlugins()...)
		case vsphere.ProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, vsphere_volume.ProbeVolumePlugins()...)
		case azure.CloudProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, azure_dd.ProbeVolumePlugins()...)
		case photon.ProviderName == cloud.ProviderName():
			allPlugins = append(allPlugins, photon_pd.ProbeVolumePlugins()...)
		}
	}

	return allPlugins
}

// NewAlphaVolumeProvisioner returns a volume provisioner to use when running in
// a cloud or development environment. The alpha implementation of provisioning
// allows 1 implied provisioner per cloud and is here only for compatibility
// with Kubernetes 1.3
// TODO: remove in Kubernetes 1.5
func NewAlphaVolumeProvisioner(cloud cloudprovider.Interface, config componentconfig.VolumeConfiguration) (volume.ProvisionableVolumePlugin, error) {
	switch {
	case !utilfeature.DefaultFeatureGate.Enabled(features.DynamicVolumeProvisioning):
		return nil, nil
	case cloud == nil && config.EnableHostPathProvisioning:
		return getProvisionablePluginFromVolumePlugins(host_path.ProbeVolumePlugins(
			volume.VolumeConfig{
				ProvisioningEnabled: true,
			}))
	case cloud != nil && aws.ProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(aws_ebs.ProbeVolumePlugins())
	case cloud != nil && gce.ProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(gce_pd.ProbeVolumePlugins())
	case cloud != nil && openstack.ProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(cinder.ProbeVolumePlugins())
	case cloud != nil && vsphere.ProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(vsphere_volume.ProbeVolumePlugins())
	case cloud != nil && azure.CloudProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(azure_dd.ProbeVolumePlugins())
	case cloud != nil && photon.ProviderName == cloud.ProviderName():
		return getProvisionablePluginFromVolumePlugins(photon_pd.ProbeVolumePlugins())
	}
	return nil, nil
}

func getProvisionablePluginFromVolumePlugins(plugins []volume.VolumePlugin) (volume.ProvisionableVolumePlugin, error) {
	for _, plugin := range plugins {
		if provisonablePlugin, ok := plugin.(volume.ProvisionableVolumePlugin); ok {
			return provisonablePlugin, nil
		}
	}
	return nil, fmt.Errorf("ProvisionablePlugin expected but not found in %#v: ", plugins)
}

// AttemptToLoadRecycler tries decoding a pod from a filepath for use as a recycler for a volume.
// If successful, this method will set the recycler on the config.
// If unsuccessful, an error is returned. Function is exported for reuse downstream.
func AttemptToLoadRecycler(path string, config *volume.VolumeConfig) error {
	if path != "" {
		recyclerPod, err := io.LoadPodFromFile(path)
		if err != nil {
			return err
		}
		if err = volume.ValidateRecyclerPodTemplate(recyclerPod); err != nil {
			return fmt.Errorf("Pod specification (%v): %v", path, err)
		}
		config.RecyclerPodTemplate = recyclerPod
	}
	return nil
}
