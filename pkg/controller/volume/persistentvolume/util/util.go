/*
Copyright 2019 The Kubernetes Authors.

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

package persistentvolume

import (
	v1 "k8s.io/api/core/v1"
	storagehelpers "k8s.io/component-helpers/storage/volume"
)

const (
	// AnnBindCompleted Annotation applies to PVCs. It indicates that the lifecycle
	// of the PVC has passed through the initial setup. This information changes how
	// we interpret some observations of the state of the objects. Value of this
	// Annotation does not matter.
	AnnBindCompleted = storagehelpers.AnnBindCompleted

	// AnnBoundByController annotation applies to PVs and PVCs. It indicates that
	// the binding (PV->PVC or PVC->PV) was installed by the controller. The
	// absence of this annotation means the binding was done by the user (i.e.
	// pre-bound). Value of this annotation does not matter.
	// External PV binders must bind PV the same way as PV controller, otherwise PV
	// controller may not handle it correctly.
	AnnBoundByController = storagehelpers.AnnBoundByController

	// AnnSelectedNode annotation is added to a PVC that has been triggered by scheduler to
	// be dynamically provisioned. Its value is the name of the selected node.
	AnnSelectedNode = storagehelpers.AnnSelectedNode

	// NotSupportedProvisioner is a special provisioner name which can be set
	// in storage class to indicate dynamic provisioning is not supported by
	// the storage.
	NotSupportedProvisioner = storagehelpers.NotSupportedProvisioner

	// AnnDynamicallyProvisioned annotation is added to a PV that has been dynamically provisioned by
	// Kubernetes. Its value is name of volume plugin that created the volume.
	// It serves both user (to show where a PV comes from) and Kubernetes (to
	// recognize dynamically provisioned PVs in its decisions).
	AnnDynamicallyProvisioned = storagehelpers.AnnDynamicallyProvisioned

	// AnnMigratedTo annotation is added to a PVC and PV that is supposed to be
	// dynamically provisioned/deleted by by its corresponding CSI driver
	// through the CSIMigration feature flags. When this annotation is set the
	// Kubernetes components will "stand-down" and the external-provisioner will
	// act on the objects
	AnnMigratedTo = storagehelpers.AnnMigratedTo

	// AnnStorageProvisioner annotation is added to a PVC that is supposed to be dynamically
	// provisioned. Its value is name of volume plugin that is supposed to provision
	// a volume for this PVC.
	AnnStorageProvisioner = storagehelpers.AnnStorageProvisioner
)

// IsDelayBindingProvisioning checks if claim provisioning with selected-node annotation
func IsDelayBindingProvisioning(claim *v1.PersistentVolumeClaim) bool {
	// When feature VolumeScheduling enabled,
	// Scheduler signal to the PV controller to start dynamic
	// provisioning by setting the "AnnSelectedNode" annotation
	// in the PVC
	_, ok := claim.Annotations[AnnSelectedNode]
	return ok
}
