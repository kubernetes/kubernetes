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

package csimigration

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume"
)

// InTreeToCSITranslator performs translation of Volume sources for PV and Volume objects
// from references to in-tree plugins to migrated CSI plugins
type InTreeToCSITranslator interface {
	IsMigratable(pv *v1.PersistentVolume, vol *v1.Volume) bool
	GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error)
	GetCSINameFromInTreeName(pluginName string) (string, error)
	IsMigratableIntreePluginByName(inTreePluginName string) bool
	TranslateInTreePVToCSI(logger klog.Logger, pv *v1.PersistentVolume) (*v1.PersistentVolume, error)
	TranslateInTreeInlineVolumeToCSI(logger klog.Logger, volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error)
}

// TranslateInTreeSpecToCSI translates a volume spec (either PV or inline volume)
// supported by an in-tree plugin to CSI
func TranslateInTreeSpecToCSI(logger klog.Logger, spec *volume.Spec, podNamespace string, translator InTreeToCSITranslator) (*volume.Spec, error) {
	var csiPV *v1.PersistentVolume
	var err error
	inlineVolume := false
	if spec.PersistentVolume != nil {
		csiPV, err = translator.TranslateInTreePVToCSI(logger, spec.PersistentVolume)
	} else if spec.Volume != nil {
		csiPV, err = translator.TranslateInTreeInlineVolumeToCSI(logger, spec.Volume, podNamespace)
		inlineVolume = true
	} else {
		err = errors.New("not a valid volume spec")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to translate in-tree pv to CSI: %v", err)
	}
	return &volume.Spec{
		Migrated:                        true,
		PersistentVolume:                csiPV,
		ReadOnly:                        spec.ReadOnly,
		InlineVolumeSpecForCSIMigration: inlineVolume,
	}, nil
}
