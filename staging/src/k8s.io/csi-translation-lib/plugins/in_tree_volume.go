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

package plugins

import (
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
)

// InTreePlugin handles translations between CSI and in-tree sources in a PV
type InTreePlugin interface {

	// TranslateInTreeStorageClassToCSI takes in-tree volume options
	// and translates them to a volume options consumable by CSI plugin
	TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error)

	// TranslateInTreeInlineVolumeToCSI takes a inline volume and will translate
	// the in-tree inline volume source to a CSIPersistentVolumeSource
	// A PV object containing the CSIPersistentVolumeSource in it's spec is returned
	TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error)

	// TranslateInTreePVToCSI takes a persistent volume and will translate
	// the in-tree pv source to a CSI Source. The input persistent volume can be modified
	TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
	// it to a in-tree Persistent Volume Source for the in-tree volume
	// by the `Driver` field in the CSI Source. The input PV object can be modified
	TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// CanSupport tests whether the plugin supports a given persistent volume
	// specification from the API.
	CanSupport(pv *v1.PersistentVolume) bool

	// CanSupportInline tests whether the plugin supports a given inline volume
	// specification from the API.
	CanSupportInline(vol *v1.Volume) bool

	// GetInTreePluginName returns the in-tree plugin name this migrates
	GetInTreePluginName() string

	// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
	GetCSIPluginName() string
}
