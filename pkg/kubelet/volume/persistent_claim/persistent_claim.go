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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/host_path"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.Plugin {
	return []volume.Plugin{&persistentClaimPlugin{nil}}
}

type persistentClaimPlugin struct {
	host volume.Host
}

var _ volume.Plugin = &persistentClaimPlugin{}

const (
	persistentClaimPluginName = "kubernetes.io/persistent-claim"
)

func (plugin *persistentClaimPlugin) Init(host volume.Host) {
	plugin.host = host
}

func (plugin *persistentClaimPlugin) Name() string {
	return persistentClaimPluginName
}

func (plugin *persistentClaimPlugin) CanSupport(spec *api.Volume) bool {

	if spec.Source.PersistentVolumeClaimVolumeSource != nil ||
		spec.Source.HostPath != nil ||
		spec.Source.AWSElasticBlockStore != nil ||
		spec.Source.GCEPersistentDisk != nil ||
		spec.Source.NFSMount != nil {
		return true
	}

	return false
}

func (plugin *persistentClaimPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {

	claimName := spec.Source.PersistentVolumeClaimVolumeSource.PersistentVolumeClaimRef.Name
	volNamespace := spec.Source.PersistentVolumeClaimVolumeSource.PersistentVolumeClaimRef.Namespace
	claim, err := plugin.host.GetKubeClient().PersistentVolumeClaims(volNamespace).Get(claimName)

	if err != nil {
		glog.V(3).Infof("Error finding claim in namespace  PersistentVolume by ClaimRef: %+v\n", spec.Source.PersistentVolumeClaimVolumeSource.PersistentVolumeClaimRef)
		return nil, err
	}

	if claim != nil {
		pv, err := plugin.host.GetKubeClient().PersistentVolumes().Get(claim.Status.VolumeRef.Name)

		if err != nil {
			glog.V(3).Infof("Error finding bound PersistentVolume by ClaimRef: %+v\n", spec.Source.PersistentVolumeClaimVolumeSource.PersistentVolumeClaimRef)
			return nil, err
		}

		wrapper := &api.Volume{
			Name:   spec.Name,
			Source: pv.Spec.Source,
		}

		volPlugin := getVolumePlugin(wrapper)

		if volPlugin == nil {
			glog.V(3).Infof("No plugin found for PersistentVolume: %+v\n", pv)
			return nil, err
		}

		volPlugin.Init(plugin.host)
		builder, err := volPlugin.NewBuilder(wrapper, podUID)

		if err != nil {
			glog.V(3).Infof("Error creating builder for ClaimRef: %+v\n", spec.Source.PersistentVolumeClaimVolumeSource.PersistentVolumeClaimRef)
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

func getVolumePlugin(spec *api.Volume) volume.Plugin {

	if spec.Source.HostPath != nil {
		return host_path.ProbeVolumePlugins()[0]
	} else if spec.Source.GCEPersistentDisk != nil {
		return gce_pd.ProbeVolumePlugins()[0]
	} else if spec.Source.AWSElasticBlockStore != nil {

		// TODO -- when this volume lands
		return nil
	} else if spec.Source.NFSMount != nil {

		// TODO -- when this volume lands
		return nil
	}
	return nil
}
