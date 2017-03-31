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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

const (
	// Default mount directory for the shm volume.
	defaultDevShmMountPath        string = "/dev/shm"
	defaultDevShmVolumeNamePrefix string = "dev-shm"
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
	if len(attributes.GetSubresource()) > 0 {
		// only run the checks below on pods proper and not subresources
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected *api.Pod but got %T", attributes.GetObject()))
	}

	// Check if the /dev/shm volume needs to be mounted on any containers.
	needDevShmVolume := false
	for _, container := range pod.Spec.InitContainers {
		devShmVolumeExists := false
		for _, volumeMount := range container.VolumeMounts {
			if volumeMount.MountPath == defaultDevShmMountPath {
				devShmVolumeExists = true
			}
		}
		if !devShmVolumeExists {
			needDevShmVolume = true
			break
		}
	}
	if !needDevShmVolume {
		for _, container := range pod.Spec.Containers {
			devShmVolumeExists := false
			for _, volumeMount := range container.VolumeMounts {
				if volumeMount.MountPath == defaultDevShmMountPath {
					devShmVolumeExists = true
				}
			}
			if !devShmVolumeExists {
				needDevShmVolume = true
				break
			}
		}
	}

	if !needDevShmVolume {
		return nil
	}
	devShmVolumeName := defaultDevShmVolumeNamePrefix + string(uuid.NewUUID())
	volume := api.Volume{
		Name: devShmVolumeName,
		VolumeSource: api.VolumeSource{
			EmptyDir: &api.EmptyDirVolumeSource{
				Medium: api.StorageMediumMemory,
			},
		},
	}
	pod.Spec.Volumes = append(pod.Spec.Volumes, volume)

	// Create the prototypical VolumeMount
	volumeMount := api.VolumeMount{
		Name:      devShmVolumeName,
		MountPath: defaultDevShmMountPath,
	}
	// Ensure every container mounts the shm volume
	for i, container := range pod.Spec.InitContainers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			if volumeMount.MountPath == defaultDevShmMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.InitContainers[i].VolumeMounts = append(pod.Spec.InitContainers[i].VolumeMounts, volumeMount)
		}
	}
	for i, container := range pod.Spec.Containers {
		existingContainerMount := false
		for _, volumeMount := range container.VolumeMounts {
			if volumeMount.MountPath == defaultDevShmMountPath {
				existingContainerMount = true
				break
			}
		}
		if !existingContainerMount {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, volumeMount)
		}
	}
	return nil
}
