/*
Copyright 2014 Google Inc. All rights reserved.

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

package persistent_claim

import (
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/host_path"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&persistentClaimPlugin{nil}}
}

type persistentClaimPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &persistentClaimPlugin{}

const (
	persistentClaimPluginName = "kubernetes.io/persistent-claim"
)

func (plugin *persistentClaimPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *persistentClaimPlugin) Name() string {
	return persistentClaimPluginName
}

func (plugin *persistentClaimPlugin) CanSupport(spec *api.Volume) bool {
	return spec.PersistentVolumeClaimVolumeSource != nil
}

func (plugin *persistentClaimPlugin) NewBuilder(spec *api.Volume, podRef *api.ObjectReference) (volume.Builder, error) {

	glog.V(5).Infof("Creating builder for %+v\n", spec)

	claimName := spec.PersistentVolumeClaimVolumeSource.ClaimName
	volNamespace := podRef.Namespace
	claim, err := plugin.host.GetKubeClient().PersistentVolumeClaims(volNamespace).Get(claimName)

	glog.V(5).Infof("Using claim: %+v\n", claim)

	if err != nil {
		glog.Errorf("Error finding claim in namespace  PersistentVolume by ClaimName: %+v\n", spec.PersistentVolumeClaimVolumeSource.ClaimName)
		return nil, err
	}

	if claim != nil {
		pv, err := plugin.host.GetKubeClient().PersistentVolumes().Get(claim.Status.VolumeRef.Name)

		if err != nil {
			glog.Errorf("Error finding bound PersistentVolume by ClaimName: %+v\n", spec.PersistentVolumeClaimVolumeSource.ClaimName)
			return nil, err
		}

		glog.V(5).Infof("Using persistent volume: %+v\n", pv)
		volPlugin := getVolumePlugin(pv.Spec)

		if volPlugin == nil {
			glog.Errorf("No plugin found for PersistentVolume: %+v\n", pv)
			return nil, err
		}

		glog.V(5).Infof("Using plugin: %s\n", volPlugin.Name())
		volSource, err := getVolumeSource((pv.Spec.PersistentVolumeSource))

		if err != nil {
			glog.Errorf("Error getting VolumeSource for %+v: \n", pv.Spec.PersistentVolumeSource)
			return nil, err
		}

		glog.V(5).Infof("Using volume source: %+v\n", volSource)
		wrapper := &api.Volume{
			Name:         spec.Name,
			VolumeSource: volSource,
		}

		volPlugin.Init(plugin.host)
		builder, err := volPlugin.NewBuilder(wrapper, podRef)

		glog.V(5).Infof("Returning builder: %s\n", builder.GetPath())

		if err != nil {
			glog.Errorf("Error creating builder for ClaimName: %+v\n", spec.PersistentVolumeClaimVolumeSource.ClaimName)
			return nil, err
		}

		if builder != nil {
			return builder, nil
		}
	}

	return nil, fmt.Errorf("No builder found for volume %+v\n", spec)
}

func (plugin *persistentClaimPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("This will never be called directly.  Volumes backing a PV have their own cleaners and Kubelet reconciles volumes in its syncLoop.")
}

func getVolumeSource(pv api.PersistentVolumeSource) (api.VolumeSource, error) {
	if pv.HostPath != nil {
		return api.VolumeSource{
			HostPath: pv.HostPath,
		}, nil
	} else if pv.GCEPersistentDisk != nil {
		return api.VolumeSource{
			GCEPersistentDisk: pv.GCEPersistentDisk,
		}, nil
	}
	return api.VolumeSource{}, fmt.Errorf("No set member found on %+v:", pv)
}

func getVolumePlugin(spec api.PersistentVolumeSpec) volume.VolumePlugin {
	if spec.HostPath != nil {
		return host_path.ProbeVolumePlugins()[0]
	} else if spec.GCEPersistentDisk != nil {
		return gce_pd.ProbeVolumePlugins()[0]
		//	} else if spec.NFS != nil {
		//		// TODO -- when this volume lands
		//		return nil
	}
	return nil
}
