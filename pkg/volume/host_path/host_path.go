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

package host_path

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
// Tests covering recycling should not use this func but instead
// use their own array of plugins w/ a custom recyclerFunc as appropriate
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&hostPathPlugin{nil, newRecycler}}
}

type hostPathPlugin struct {
	host volume.VolumeHost
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error)
}

var _ volume.VolumePlugin = &hostPathPlugin{}
var _ volume.PersistentVolumePlugin = &hostPathPlugin{}
var _ volume.RecyclableVolumePlugin = &hostPathPlugin{}

const (
	hostPathPluginName = "kubernetes.io/host-path"
)

func (plugin *hostPathPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *hostPathPlugin) Name() string {
	return hostPathPluginName
}

func (plugin *hostPathPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.VolumeSource.HostPath != nil || spec.PersistentVolumeSource.HostPath != nil
}

func (plugin *hostPathPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *hostPathPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, _ mount.Interface) (volume.Builder, error) {
	if spec.VolumeSource.HostPath != nil {
		return &hostPath{spec.VolumeSource.HostPath.Path}, nil
	} else {
		return &hostPath{spec.PersistentVolumeSource.HostPath.Path}, nil
	}
}

func (plugin *hostPathPlugin) NewCleaner(volName string, podUID types.UID, _ mount.Interface) (volume.Cleaner, error) {
	return &hostPath{""}, nil
}

func (plugin *hostPathPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.host)
}

func newRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	if spec.VolumeSource.HostPath != nil {
		return &hostPathRecycler{spec.Name, spec.VolumeSource.HostPath.Path, host}, nil
	} else {
		return &hostPathRecycler{spec.Name, spec.PersistentVolumeSource.HostPath.Path, host}, nil
	}
}

// HostPath volumes represent a bare host file or directory mount.
// The direct at the specified path will be directly exposed to the container.
type hostPath struct {
	path string
}

// SetUp does nothing.
func (hp *hostPath) SetUp() error {
	return nil
}

// SetUpAt does not make sense for host paths - probably programmer error.
func (hp *hostPath) SetUpAt(dir string) error {
	return fmt.Errorf("SetUpAt() does not make sense for host paths")
}

func (hp *hostPath) GetPath() string {
	return hp.path
}

// TearDown does nothing.
func (hp *hostPath) TearDown() error {
	return nil
}

// TearDownAt does not make sense for host paths - probably programmer error.
func (hp *hostPath) TearDownAt(dir string) error {
	return fmt.Errorf("TearDownAt() does not make sense for host paths")
}

// hostPathRecycler scrubs a hostPath volume by running "rm -rf" on the volume in a pod
// This recycler only works on a single host cluster and is for testing purposes only.
type hostPathRecycler struct {
	name string
	path string
	host volume.VolumeHost
}

func (r *hostPathRecycler) GetPath() string {
	return r.path
}

// Recycler provides methods to reclaim the volume resource.
func (r *hostPathRecycler) Recycle() error {
	uuid := string(util.NewUUID())
	timeout := int64(30)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "scrubber-" + r.name,
			Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"scrubber": uuid,
			},
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: uuid,
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{ r.path },
					},
				},
			},
			Containers: []api.Container{
				{
					Name: "scrubber-" + uuid,
					Image: "busybox",
					Command: []string{"ls -la"},
					WorkingDir: "/scrub",
					VolumeMounts: []api.VolumeMount{
						{
							Name: uuid,
							MountPath: "/scrub",
						},
					},
				},
			},
			ActiveDeadlineSeconds: &timeout,
		},
	}
	return volume.ScrubPodVolumeAndWatchUntilCompletion(pod, r.host.GetKubeClient())
}
