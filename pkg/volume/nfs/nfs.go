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

package nfs

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/network_volume"
)

// This is the primary entrypoint for volume plugins.
// Tests covering recycling should not use this func but instead
// use their own array of plugins w/ a custom recyclerFunc as appropriate
func ProbeVolumePlugins() []volume.VolumePlugin {
	plugins := []volume.VolumePlugin{
		&nfsPlugin{
			NetworkVolumePlugin: &network_volume.NetworkVolumePlugin{
				PluginName: nfsPluginName},
			newRecyclerFunc: newRecycler},
	}
	return plugins
}

type nfsPlugin struct {
	*network_volume.NetworkVolumePlugin
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error)
}

var _ volume.VolumePlugin = &nfsPlugin{}
var _ volume.PersistentVolumePlugin = &nfsPlugin{}
var _ volume.RecyclableVolumePlugin = &nfsPlugin{}

const (
	nfsPluginName = "kubernetes.io/nfs"
)

func (plugin *nfsPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.VolumeSource.NFS != nil || spec.PersistentVolumeSource.NFS != nil
}

func (plugin *nfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *nfsPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod, mounter)
}

func (plugin *nfsPlugin) newBuilderInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface) (volume.Builder, error) {
	var source *api.NFSVolumeSource

	if spec.VolumeSource.NFS != nil {
		source = spec.VolumeSource.NFS
	} else {
		source = spec.PersistentVolumeSource.NFS
	}
	return &nfs{
		NetworkVolume: &network_volume.NetworkVolume{
			VolName: spec.Name,
			Plugin:  plugin,
			PodUID:  pod.UID,
			Mounter: &nfsNetworkVolumeMounter{
				Interface:  mounter,
				server:     source.Server,
				exportPath: source.Path,
				readOnly:   source.ReadOnly,
			},
		},
	}, nil

}

func (plugin *nfsPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, mounter)
}

func (plugin *nfsPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &nfs{
		NetworkVolume: &network_volume.NetworkVolume{
			VolName: volName,
			Plugin:  plugin,
			PodUID:  podUID,
			Mounter: &nfsNetworkVolumeMounter{
				Interface:  mounter,
				server:     "",
				exportPath: "",
				readOnly:   false,
			},
		},
	}, nil
}

func (plugin *nfsPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.GetHost())
}

func newRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	if spec.VolumeSource.HostPath != nil {
		return &nfsRecycler{
			name:   spec.Name,
			server: spec.VolumeSource.NFS.Server,
			path:   spec.VolumeSource.NFS.Path,
			host:   host,
		}, nil
	} else {
		return &nfsRecycler{
			name:   spec.Name,
			server: spec.PersistentVolumeSource.NFS.Server,
			path:   spec.PersistentVolumeSource.NFS.Path,
			host:   host,
		}, nil
	}
}

// NFS volumes represent a bare host file or directory mount of an NFS export.
type nfs struct {
	*network_volume.NetworkVolume
}

type nfsNetworkVolumeMounter struct {
	mount.Interface
	server     string
	exportPath string
	readOnly   bool
	plugin     *nfsPlugin
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error)
}

func (mounter *nfsNetworkVolumeMounter) MountVolume(dest string) error {
	source := fmt.Sprintf("%s:%s", mounter.server, mounter.exportPath)
	options := []string{}
	if mounter.readOnly {
		options = append(options, "ro")
	}
	return mounter.Mount(source, dest, "nfs", options)
}

func (mounter *nfsNetworkVolumeMounter) UnmountVolume(dir string) error {
	// Unmount dir
	return mounter.Unmount(dir)
}

func (mounter *nfsNetworkVolumeMounter) AttachVolume() error {
	return nil
}

func (mounter *nfsNetworkVolumeMounter) DetachVolume(path string) error {
	return nil
}

func (mounter *nfsNetworkVolumeMounter) GetGlobalPath() string {
	return ""
}

// nfsRecycler scrubs an NFS volume by running "rm -rf" on the volume in a pod.
type nfsRecycler struct {
	name   string
	server string
	path   string
	host   volume.VolumeHost
}

func (r *nfsRecycler) GetPath() string {
	return r.path
}

// Recycler provides methods to reclaim the volume resource.
// A NFS volume is recycled by scheduling a pod to run "rm -rf" on the contents of the volume.
// Recycle blocks until the pod has completed or any error occurs.
// The scrubber pod's is expected to succeed within 5 minutes else an error will be returned
func (r *nfsRecycler) Recycle() error {
	timeout := int64(300) // 5 minutes
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-scrubber-" + util.ShortenString(r.name, 44) + "-",
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			ActiveDeadlineSeconds: &timeout,
			RestartPolicy:         api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: "vol",
					VolumeSource: api.VolumeSource{
						NFS: &api.NFSVolumeSource{
							Server: r.server,
							Path:   r.path,
						},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "scrubber",
					Image: "gcr.io/google_containers/busybox",
					// delete the contents of the volume, but not the directory itself
					Command: []string{"/bin/sh"},
					// the scrubber:
					//		1. validates the /scrub directory exists
					// 		2. creates a text file to be scrubbed
					//		3. performs rm -rf on the directory
					//		4. tests to see if the directory is empty
					// the pod fails if the error code is returned
					Args: []string{"-c", "test -e /scrub && echo $(date) > /scrub/trash.txt && rm -rf /scrub/* && test -z \"$(ls -A /scrub)\" || exit 1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "vol",
							MountPath: "/scrub",
						},
					},
				},
			},
		},
	}
	return volume.ScrubPodVolumeAndWatchUntilCompletion(pod, r.host.GetKubeClient())
}
