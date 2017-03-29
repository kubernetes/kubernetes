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

package devshm

import (
	"fmt"
	"io"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

const (
	// Default name for the devShm volume.
	defaultDevShmVolumeName string = "dev-shm"
	// Default mount directory for the shm volume.
	defaultDevShmMountPath string = "/dev/shm"
)

func init() {
	admission.RegisterPlugin("DefaultDevShm", func(config io.Reader) (admission.Interface, error) {
		return NewDefaultDevShm(), nil
	})
}

// plugin contains the client used by the admission controller
// It will add default /dev/shm volume to all pods.
type plugin struct {
	*admission.Handler
}

// NewDefaultTolerationSeconds creates a new instance of the DefaultTolerationSeconds admission controller
func NewDefaultDevShm() admission.Interface {
	return &plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

func (p *plugin) Admit(attributes admission.Attributes) (err error) {
	if attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}
	glog.Errorf("vishh:1")
	if len(attributes.GetSubresource()) > 0 {
		// only run the checks below on pods proper and not subresources
		return nil
	}
	glog.Errorf("vishh:2")
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected *api.Pod but got %T", attributes.GetObject()))
	}

	// Find the volume and volume name for the ServiceAccountTokenSecret if it already exists
	hasDevShmVolume := false
	glog.Errorf("%+v", pod.Spec.Volumes)
	for _, volume := range pod.Spec.Volumes {
		if volume.EmptyDir != nil && volume.EmptyDir.Medium == api.StorageMediumMemory && volume.Name == defaultDevShmVolumeName {
			hasDevShmVolume = true
			break
		}
	}
	glog.Errorf("vishh:3")
	// Create the prototypical VolumeMount
	volumeMount := api.VolumeMount{
		Name:      defaultDevShmVolumeName,
		MountPath: defaultDevShmMountPath,
	}
	// Ensure every container mounts the shm volume
	needsDevShmVolume := false
	for i, container := range pod.Spec.InitContainers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			// Existing mounts at the default mount path prevent mounting of the API token
			if volumeMount.MountPath == defaultDevShmMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.InitContainers[i].VolumeMounts = append(pod.Spec.InitContainers[i].VolumeMounts, volumeMount)
			needsDevShmVolume = true
			glog.Errorf("%+v", pod.Spec.InitContainers[i].VolumeMounts)
		}
	}
	for i, container := range pod.Spec.Containers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			// Existing mounts at the default mount path prevent mounting of the API token
			if volumeMount.MountPath == defaultDevShmMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, volumeMount)
			glog.Errorf("%+v", pod.Spec.Containers[i].VolumeMounts)
			needsDevShmVolume = true
		}
	}
	// Add the volume if a container needs it
	if !hasDevShmVolume && needsDevShmVolume {
		volume := api.Volume{
			Name: defaultDevShmVolumeName,
			VolumeSource: api.VolumeSource{
				EmptyDir: &api.EmptyDirVolumeSource{
					Medium: api.StorageMediumMemory,
				},
			},
		}
		pod.Spec.Volumes = append(pod.Spec.Volumes, volume)
		glog.Errorf("vishh:4")
	}
	glog.Errorf("vishh:5")
	return nil
}
