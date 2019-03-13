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

import "k8s.io/api/core/v1"

// InTreePlugin handles translations between CSI and in-tree sources in a PV
type InTreePlugin interface {

	// TranslateInTreeStorageClassParametersToCSI takes in-tree storage class
	// parameters and translates them to a set of parameters consumable by CSI plugin
	TranslateInTreeStorageClassParametersToCSI(scParameters map[string]string) (map[string]string, error)

	// TranslateInTreePVToCSI takes a persistent volume and will translate
	// the in-tree source to a CSI Source. The input persistent volume can be modified
	TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
	// it to a in-tree Persistent Volume Source for the in-tree volume
	// by the `Driver` field in the CSI Source. The input PV object can be modified
	TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// CanSupport tests whether the plugin supports a given volume
	// specification from the API.
	CanSupport(pv *v1.PersistentVolume) bool

	// GetInTreePluginName returns the in-tree plugin name this migrates
	GetInTreePluginName() string

	// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
	GetCSIPluginName() string
}
