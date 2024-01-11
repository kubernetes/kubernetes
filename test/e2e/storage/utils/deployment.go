/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"path"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime"
	e2eframework "k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// PatchCSIDeployment modifies the CSI driver deployment:
// - replaces the provisioner name
// - forces pods onto a specific host
//
// All of that is optional, see PatchCSIOptions. Just beware
// that not renaming the CSI driver deployment can be problematic:
//   - when multiple tests deploy the driver, they need
//     to run sequentially
//   - might conflict with manual deployments
//
// This function is written so that it works for CSI driver deployments
// that follow these conventions:
//   - driver and provisioner names are identical
//   - the driver binary accepts a --drivername parameter
//   - the paths inside the container are either fixed
//     and don't need to be patch (for example, --csi-address=/csi/csi.sock is
//     okay) or are specified directly in a parameter (for example,
//     --kubelet-registration-path=/var/lib/kubelet/plugins/csi-hostpath/csi.sock)
//
// Driver deployments that are different will have to do the patching
// without this function, or skip patching entirely.
func PatchCSIDeployment(tCtx ktesting.TContext, o PatchCSIOptions, object runtime.Object) {
	rename := o.OldDriverName != "" && o.NewDriverName != "" &&
		o.OldDriverName != o.NewDriverName

	substKubeletRootDir := func(s string) string {
		return strings.ReplaceAll(s, "/var/lib/kubelet/", e2eframework.TestContext.KubeletRootDir+"/")
	}

	patchVolumes := func(volumes []v1.Volume) {
		if !rename {
			return
		}
		for i := range volumes {
			volume := &volumes[i]
			if volume.HostPath != nil {
				// Update paths like /var/lib/kubelet/plugins/<provisioner>.
				p := &volume.HostPath.Path
				dir, file := path.Split(*p)
				if file == o.OldDriverName {
					*p = path.Join(dir, o.NewDriverName)
				}
				// Inject non-standard kubelet path.
				*p = substKubeletRootDir(*p)
			}
		}
	}

	patchContainers := func(containers []v1.Container) {
		for i := range containers {
			container := &containers[i]
			if rename {
				for e := range container.Args {
					// Inject test-specific provider name into paths like this one:
					// --kubelet-registration-path=/var/lib/kubelet/plugins/csi-hostpath/csi.sock
					container.Args[e] = strings.Replace(container.Args[e], "/"+o.OldDriverName+"/", "/"+o.NewDriverName+"/", 1)
				}
			}

			// Modify --kubelet-registration-path.
			for e := range container.Args {
				container.Args[e] = substKubeletRootDir(container.Args[e])
			}
			for e := range container.VolumeMounts {
				container.VolumeMounts[e].MountPath = substKubeletRootDir(container.VolumeMounts[e].MountPath)
			}

			if len(o.Features) > 0 && len(o.Features[container.Name]) > 0 {
				featuregateString := strings.Join(o.Features[container.Name], ",")
				container.Args = append(container.Args, fmt.Sprintf("--feature-gates=%s", featuregateString))
			}

			// Overwrite driver name resp. provider name
			// by appending a parameter with the right
			// value.
			switch container.Name {
			case o.DriverContainerName:
				container.Args = append(container.Args, o.DriverContainerArguments...)
			}
		}
	}

	patchPodSpec := func(spec *v1.PodSpec) {
		patchContainers(spec.Containers)
		patchVolumes(spec.Volumes)
		if o.NodeName != "" {
			e2epod.SetNodeSelection(spec, e2epod.NodeSelection{Name: o.NodeName})
		}
	}

	switch object := object.(type) {
	case *appsv1.ReplicaSet:
		patchPodSpec(&object.Spec.Template.Spec)
	case *appsv1.DaemonSet:
		patchPodSpec(&object.Spec.Template.Spec)
	case *appsv1.StatefulSet:
		patchPodSpec(&object.Spec.Template.Spec)
	case *appsv1.Deployment:
		patchPodSpec(&object.Spec.Template.Spec)
	case *storagev1.StorageClass:
		if o.NewDriverName != "" {
			// Driver name is expected to be the same
			// as the provisioner name here.
			object.Provisioner = o.NewDriverName
		}
	case *storagev1.CSIDriver:
		if o.NewDriverName != "" {
			object.Name = o.NewDriverName
		}
		if o.PodInfo != nil {
			object.Spec.PodInfoOnMount = o.PodInfo
		}
		if o.StorageCapacity != nil {
			object.Spec.StorageCapacity = o.StorageCapacity
		}
		if o.CanAttach != nil {
			object.Spec.AttachRequired = o.CanAttach
		}
		if o.VolumeLifecycleModes != nil {
			object.Spec.VolumeLifecycleModes = *o.VolumeLifecycleModes
		}
		if o.TokenRequests != nil {
			object.Spec.TokenRequests = o.TokenRequests
		}
		if o.RequiresRepublish != nil {
			object.Spec.RequiresRepublish = o.RequiresRepublish
		}
		if o.FSGroupPolicy != nil {
			object.Spec.FSGroupPolicy = o.FSGroupPolicy
		}
		if o.SELinuxMount != nil {
			object.Spec.SELinuxMount = o.SELinuxMount
		}
	}
}

// PatchCSIOptions controls how PatchCSIDeployment patches the objects.
type PatchCSIOptions struct {
	// The original driver name.
	OldDriverName string
	// The driver name that replaces the original name.
	// Can be empty (not used at all) or equal to OldDriverName
	// (then it will be added were appropriate without renaming
	// in existing fields).
	NewDriverName string
	// The name of the container which has the CSI driver binary.
	// If non-empty, DriverContainerArguments are added to argument
	// list in container with that name.
	DriverContainerName string
	// List of arguments to add to container with
	// DriverContainerName.
	DriverContainerArguments []string
	// The name of the container which has the provisioner binary.
	// If non-empty, --provisioner with new name will be appended
	// to the argument list.
	ProvisionerContainerName string
	// The name of the container which has the snapshotter binary.
	// If non-empty, --snapshotter with new name will be appended
	// to the argument list.
	SnapshotterContainerName string
	// If non-empty, all pods are forced to run on this node.
	NodeName string
	// If not nil, the value to use for the CSIDriver.Spec.PodInfo
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	PodInfo *bool
	// If not nil, the value to use for the CSIDriver.Spec.CanAttach
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	CanAttach *bool
	// If not nil, the value to use for the CSIDriver.Spec.StorageCapacity
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	StorageCapacity *bool
	// If not nil, the value to use for the CSIDriver.Spec.VolumeLifecycleModes
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	VolumeLifecycleModes *[]storagev1.VolumeLifecycleMode
	// If not nil, the value to use for the CSIDriver.Spec.TokenRequests
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	TokenRequests []storagev1.TokenRequest
	// If not nil, the value to use for the CSIDriver.Spec.RequiresRepublish
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	RequiresRepublish *bool
	// If not nil, the value to use for the CSIDriver.Spec.FSGroupPolicy
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	FSGroupPolicy *storagev1.FSGroupPolicy
	// If not nil, the value to use for the CSIDriver.Spec.SELinuxMount
	// field *if* the driver deploys a CSIDriver object. Ignored
	// otherwise.
	SELinuxMount *bool
	// If not nil, the values will be used for setting feature arguments to
	// specific sidecar.
	// Feature is a map - where key is sidecar name such as:
	//	-- key: resizer
	//	-- value: []string{feature-gates}
	Features map[string][]string
}
