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

package image

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

// imagePlugin is the image volume plugin which acts as a stub to provide the
// functionality the volume manager expects. The real volume source
// implementation is part of the kubelet code and gated by the Kubernetes
// feature "ImageVolume"
// See: https://kep.k8s.io/4639
type imagePlugin struct {
	spec *volume.Spec
	volume.MetricsNil
}

var _ volume.VolumePlugin = &imagePlugin{}
var _ volume.Mounter = &imagePlugin{}
var _ volume.Unmounter = &imagePlugin{}
var _ volume.Volume = &imagePlugin{}

const pluginName = "kubernetes.io/image"

func ProbeVolumePlugins() []volume.VolumePlugin {
	p := &imagePlugin{}
	return []volume.VolumePlugin{p}
}

func (o *imagePlugin) Init(volume.VolumeHost) error                    { return nil }
func (o *imagePlugin) GetPluginName() string                           { return pluginName }
func (o *imagePlugin) GetVolumeName(spec *volume.Spec) (string, error) { return o.spec.Name(), nil }

func (o *imagePlugin) CanSupport(spec *volume.Spec) bool {
	return spec != nil && spec.Volume != nil && spec.Volume.Image != nil
}

func (o *imagePlugin) NewMounter(spec *volume.Spec, podRef *v1.Pod) (volume.Mounter, error) {
	return o, nil
}

func (o *imagePlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return o, nil
}

func (o *imagePlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	return volume.ReconstructedVolume{Spec: o.spec}, nil
}

func (o *imagePlugin) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       true,
		Managed:        true,
		SELinuxRelabel: true,
	}
}

func (o *imagePlugin) GetPath() string                                             { return "" }
func (o *imagePlugin) RequiresFSResize() bool                                      { return false }
func (o *imagePlugin) RequiresRemount(spec *volume.Spec) bool                      { return false }
func (o *imagePlugin) SetUp(mounterArgs volume.MounterArgs) error                  { return nil }
func (o *imagePlugin) SetUpAt(dir string, mounterArgs volume.MounterArgs) error    { return nil }
func (o *imagePlugin) SupportsMountOption() bool                                   { return false }
func (o *imagePlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) { return false, nil }
func (o *imagePlugin) TearDown() error                                             { return nil }
func (o *imagePlugin) TearDownAt(string) error                                     { return nil }
