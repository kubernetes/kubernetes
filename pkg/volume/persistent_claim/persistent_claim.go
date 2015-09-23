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

package persistent_claim

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&persistentClaimPlugin{host: nil}}
}

type persistentClaimPlugin struct {
	host     volume.VolumeHost
	readOnly bool
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

func (plugin *persistentClaimPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.PersistentVolumeClaim != nil
}

func (plugin *persistentClaimPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	claim, err := plugin.host.GetKubeClient().PersistentVolumeClaims(pod.Namespace).Get(spec.Volume.PersistentVolumeClaim.ClaimName)
	if err != nil {
		glog.Errorf("Error finding claim: %+v\n", spec.Volume.PersistentVolumeClaim.ClaimName)
		return nil, err
	}

	if claim.Spec.VolumeName == "" {
		return nil, fmt.Errorf("The claim %+v is not yet bound to a volume", claim.Name)
	}

	pv, err := plugin.host.GetKubeClient().PersistentVolumes().Get(claim.Spec.VolumeName)
	if err != nil {
		glog.Errorf("Error finding persistent volume for claim: %+v\n", claim.Name)
		return nil, err
	}

	if pv.Spec.ClaimRef == nil {
		glog.Errorf("The volume is not yet bound to the claim. Expected to find the bind on volume.Spec.ClaimRef: %+v", pv)
		return nil, err
	}

	if pv.Spec.ClaimRef.UID != claim.UID {
		glog.Errorf("Expected volume.Spec.ClaimRef.UID %+v but have %+v", pv.Spec.ClaimRef.UID, claim.UID)
		return nil, err
	}

	builder, err := plugin.host.NewWrapperBuilder(volume.NewSpecFromPersistentVolume(pv, spec.ReadOnly), pod, opts)
	if err != nil {
		glog.Errorf("Error creating builder for claim: %+v\n", claim.Name)
		return nil, err
	}

	return builder, nil
}

func (plugin *persistentClaimPlugin) IsReadOnly() bool {
	return plugin.readOnly
}

func (plugin *persistentClaimPlugin) NewCleaner(_ string, _ types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("This will never be called directly. The PV backing this claim has a cleaner.  Kubelet uses that cleaner, not this one, when removing orphaned volumes.")
}
