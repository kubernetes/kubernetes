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

package validation

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/aws_ebs"
	"k8s.io/kubernetes/pkg/volume/azure_dd"
	"k8s.io/kubernetes/pkg/volume/azure_file"
	"k8s.io/kubernetes/pkg/volume/cephfs"
	"k8s.io/kubernetes/pkg/volume/cinder"
	"k8s.io/kubernetes/pkg/volume/configmap"
	"k8s.io/kubernetes/pkg/volume/downwardapi"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	"k8s.io/kubernetes/pkg/volume/fc"
	"k8s.io/kubernetes/pkg/volume/flexvolume"
	"k8s.io/kubernetes/pkg/volume/flocker"
	"k8s.io/kubernetes/pkg/volume/gce_pd"
	"k8s.io/kubernetes/pkg/volume/git_repo"
	"k8s.io/kubernetes/pkg/volume/glusterfs"
	"k8s.io/kubernetes/pkg/volume/host_path"
	"k8s.io/kubernetes/pkg/volume/iscsi"
	"k8s.io/kubernetes/pkg/volume/nfs"
	"k8s.io/kubernetes/pkg/volume/photon_pd"
	"k8s.io/kubernetes/pkg/volume/projected"
	"k8s.io/kubernetes/pkg/volume/quobyte"
	"k8s.io/kubernetes/pkg/volume/rbd"
	"k8s.io/kubernetes/pkg/volume/secret"
	"k8s.io/kubernetes/pkg/volume/vsphere_volume"
)

func probeVolumePlugins() []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}

	// list of volume plugins to probe for
	allPlugins = append(allPlugins, aws_ebs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, empty_dir.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, git_repo.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, secret.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, iscsi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, glusterfs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, rbd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, cinder.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, quobyte.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, cephfs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, downwardapi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, fc.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, flocker.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, flexvolume.ProbeVolumePlugins("")...)
	allPlugins = append(allPlugins, azure_file.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, configmap.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, vsphere_volume.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, azure_dd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, photon_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, projected.ProbeVolumePlugins()...)
	return allPlugins
}

func findPluginBySpec(volumePlugins []volume.VolumePlugin, pv *api.PersistentVolume) volume.VolumePlugin {
	matches := []volume.VolumePlugin{}
	v1Pv := &v1.PersistentVolume{}
	err := v1.Convert_api_PersistentVolume_To_v1_PersistentVolume(pv, v1Pv, nil)
	if err != nil {
		glog.Errorf("Error converting to v1.PersistentVolume: %v", err)
		return nil
	}
	volumeSpec := &volume.Spec{PersistentVolume: v1Pv}
	for _, plugin := range volumePlugins {
		if plugin.CanSupport(volumeSpec) {
			matches = append(matches, plugin)
		}
	}

	if len(matches) == 0 {
		glog.V(5).Infof("No matching plugin found for : %s", pv.Name)
		return nil
	}

	if len(matches) > 1 {
		glog.V(3).Infof("multiple volume plugins matched for : %s ", pv.Name)
		return nil
	}

	return matches[0]
}
