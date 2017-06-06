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

package rook

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type rookAttacher struct {
	host volume.VolumeHost
}

const (
	tprGroup              = "rook.io"
	tprVersion            = "v1"
	tprKind               = "Volumeattach"
	volumeAttachConfigMap = "rook-volume-attach"
	devicePathKey         = "devicePath"
	mountOptionsKey       = "mountOptions"
	checkSleepDuration    = time.Second
)

var _ volume.Attacher = &rookAttacher{}

// Attach maps a rook volume to the host and returns the attachment ID.
func (attacher *rookAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	// Create a VolumeAttach TPR instance
	tprName := generateTPRName(volumeSource.Cluster, volumeSource.VolumeGroup, volumeSource.VolumeID, string(nodeName))
	volumeAttach := &VolumeAttach{
		ObjectMeta: metav1.ObjectMeta{
			Name: tprName,
		},
		Spec: VolumeAttachSpec{
			VolumeID:    volumeSource.VolumeID,
			VolumeGroup: volumeSource.VolumeGroup,
			Node:        string(nodeName),
		},
		Status: VolumeAttachStatus{
			State:   VolumeAttachStatePending,
			Message: "Created Volume Attach TPR. Not processed yet",
		},
	}
	volumeAttach.APIVersion = fmt.Sprintf("%s/%s", tprGroup, tprVersion)
	volumeAttach.Kind = tprKind

	var result VolumeAttach
	body, _ := json.Marshal(volumeAttach)
	uri := fmt.Sprintf("apis/%s/%s/namespaces/%s/%s", tprGroup, tprVersion, volumeSource.Cluster, attachmentPluralResources)

	glog.V(4).Infof("Rook: Creating TPR %s.", body)
	err = attacher.host.GetKubeClient().Core().RESTClient().Post().
		RequestURI(uri).
		Body(body).
		Do().Into(&result)
	if err != nil {
		if !errors.IsAlreadyExists(err) {
			glog.Errorf("Rook: Failed to create VolumeAttach TPR: %+v", err)
			return "", err
		}
	}
	glog.Infof("Rook: TPR %s created successfully.", tprName)
	return tprName, nil
}

func (attacher *rookAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	// Fetch a list of VolumeAttach TPRs
	volumeAttachTPRs := VolumeAttachList{}
	uri := fmt.Sprintf("apis/%s/%s/%s", tprGroup, tprVersion, attachmentPluralResources)
	err := attacher.host.GetKubeClient().Core().RESTClient().Get().
		RequestURI(uri).
		Do().
		Into(&volumeAttachTPRs)
	if err != nil {
		glog.Errorf("Rook: Failed to get list of VolumeAttach TPR: %+v", err)
		return nil, err
	}

	// Gather up all TPRs for this node.
	volumesAttached := make(map[string]bool)
	for _, volumeAttach := range volumeAttachTPRs.VolumeAttachs {
		if volumeAttach.Spec.Node == string(nodeName) {
			volumesAttached[volumeAttach.Name] = true
		}
	}

	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			glog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}
		tprName := generateTPRName(volumeSource.Cluster, volumeSource.VolumeGroup, volumeSource.VolumeID, string(nodeName))
		_, ok := volumesAttached[tprName]
		volumesAttachedCheck[spec] = ok
	}
	return volumesAttachedCheck, nil
}

func (attacher *rookAttacher) WaitForAttach(spec *volume.Spec, tprName string, timeout time.Duration) (string, error) {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Rook: Fetching TPR %s.", tprName)
			volumeAttachTPR := VolumeAttach{}
			uri := fmt.Sprintf("apis/%s/%s/namespaces/%s/%s", tprGroup, tprVersion, volumeSource.Cluster, attachmentPluralResources)
			err = attacher.host.GetKubeClient().Core().RESTClient().Get().
				RequestURI(uri).
				Name(tprName).
				Do().
				Into(&volumeAttachTPR)
			if err != nil {
				return "", fmt.Errorf("Rook: Unable to get TPR %s: %v", tprName, err)
			}
			if volumeAttachTPR.Status.State != VolumeAttachStatePending {
				if volumeAttachTPR.Status.State == VolumeAttachStateAttached {
					configmaps, err := attacher.host.GetKubeClient().Core().ConfigMaps(volumeSource.Cluster).Get(volumeAttachConfigMap, metav1.GetOptions{})
					if err != nil {
						return "", fmt.Errorf("Rook: Unable to get configmap %s: %v", volumeAttachConfigMap, err)
					}
					return configmaps.Data[fmt.Sprintf("%s.%s", tprName, devicePathKey)], nil
				}
				return "", fmt.Errorf("Rook: Volume Attached TPR %s failed: %s", tprName, volumeAttachTPR.Status.Message)
			}
		case <-timer.C:
			return "", fmt.Errorf("Rook: Could not attach volume %s/%s TPR %s. Timeout waiting for volume attach", volumeSource.VolumeGroup, volumeSource.VolumeID, tprName)
		}
	}
}

func (attacher *rookAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	devName := generateVolumeName(volumeSource.Cluster, volumeSource.VolumeGroup, volumeSource.VolumeID)
	return makeGlobalPDName(attacher.host, devName), nil
}

func (attacher *rookAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		mountOptions := volume.MountOptionFromSpec(spec, options...)
		mountOptions = volume.JoinMountOptions(mountOptions, options)

		// Get extra mounting option from the rook configmap (if any)
		configmaps, err := attacher.host.GetKubeClient().Core().ConfigMaps(volumeSource.Cluster).Get(volumeAttachConfigMap, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("Rook: Unable to get configmap %s: %v", volumeAttachConfigMap, err)
		}

		tprName := generateTPRName(volumeSource.Cluster, volumeSource.VolumeGroup, volumeSource.VolumeID, attacher.host.GetHostName())
		extraOptions := configmaps.Data[fmt.Sprintf("%s.%s", tprName, mountOptionsKey)]
		if extraOptions != "" {
			mountOptions = volume.JoinMountOptions(strings.Split(",", extraOptions), mountOptions)
		}

		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		glog.V(4).Infof("Rook: Formatting spec %v devicePath %v deviceMountPath %v fs %v with options %+v", spec.Name(), devicePath, deviceMountPath, volumeSource.FSType, options)
	}
	return nil
}

type rookDetacher struct {
	host volume.VolumeHost
}

var _ volume.Detacher = &rookDetacher{}

// Detach unmaps a rook image from the host
func (detacher *rookDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	// volumeName is in the form of cluster-group-id
	names := strings.Split(volumeName, "-")
	namespace := names[0]
	group := names[1]
	id := names[2]
	tprName := generateTPRName(namespace, group, id, string(nodeName))

	glog.Infof("Rook: Deleting TPR %s from namespace %s", tprName, namespace)
	uri := fmt.Sprintf("apis/%s/%s/namespaces/%s/%s", tprGroup, tprVersion, namespace, attachmentPluralResources)
	return detacher.host.GetKubeClient().Core().RESTClient().Delete().
		RequestURI(uri).
		Name(tprName).
		Do().
		Error()
}

func (detacher *rookDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.host.GetMounter())
}

func generateTPRName(cluster, group, id, nodeName string) string {
	tprName := fmt.Sprintf("attach-%s-%s-%s-%s", cluster, group, id, nodeName)
	return strings.ToLower(strings.Replace(tprName, ".", "-", -1))
}
