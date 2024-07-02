/*
Copyright 2024 The Kubernetes Authors.

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

package oci

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

// ociPlugin is the OCI volume plugin which acts as a stub to provide the
// functionality the volume manager expects. The real volume source
// implementation is part of the kubelet code and gated by the Kubernetes
// feature "OCIVolumeSource"
// See: https://kep.k8s.io/4639
type ociPlugin struct {
	spec *volume.Spec
	volume.MetricsNil
}

var _ volume.VolumePlugin = &ociPlugin{}

const pluginName = "kubernetes.io/oci"

func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &ociPlugin{}
	return []volume.VolumePlugin{p}
}

func (o *ociPlugin) Init(volume.VolumeHost) error                    { return nil }
func (o *ociPlugin) GetPluginName() string                           { return pluginName }
func (o *ociPlugin) GetVolumeName(spec *volume.Spec) (string, error) { return o.spec.Name(), nil }

func (o *ociPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume.OCI != nil
}

func (o *ociPlugin) NewMounter(spec *volume.Spec, podRef *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return o, nil
}

func (o *ociPlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return o, nil
}

func (o *ociPlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	return volume.ReconstructedVolume{Spec: o.spec}, nil
}

func (o *ociPlugin) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       true,
		Managed:        true,
		SELinuxRelabel: true,
	}
}

func (o *ociPlugin) GetPath() string                                             { return "" }
func (o *ociPlugin) RequiresFSResize() bool                                      { return false }
func (o *ociPlugin) RequiresRemount(spec *volume.Spec) bool                      { return false }
func (o *ociPlugin) SetUp(mounterArgs volume.MounterArgs) error                  { return nil }
func (o *ociPlugin) SetUpAt(dir string, mounterArgs volume.MounterArgs) error    { return nil }
func (o *ociPlugin) SupportsMountOption() bool                                   { return false }
func (o *ociPlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) { return false, nil }
func (o *ociPlugin) TearDown() error                                             { return nil }
func (o *ociPlugin) TearDownAt(string) error                                     { return nil }
