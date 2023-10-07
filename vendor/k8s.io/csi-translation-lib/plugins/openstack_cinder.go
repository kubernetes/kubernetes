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
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// CinderDriverName is the name of the CSI driver for Cinder
	CinderDriverName = "cinder.csi.openstack.org"
	// CinderTopologyKey is the zonal topology key for Cinder CSI Driver
	CinderTopologyKey = "topology.cinder.csi.openstack.org/zone"
	// CinderInTreePluginName is the name of the intree plugin for Cinder
	CinderInTreePluginName = "kubernetes.io/cinder"
)

var _ InTreePlugin = (*osCinderCSITranslator)(nil)

// osCinderCSITranslator handles translation of PV spec from In-tree Cinder to CSI Cinder and vice versa
type osCinderCSITranslator struct{}

// NewOpenStackCinderCSITranslator returns a new instance of osCinderCSITranslator
func NewOpenStackCinderCSITranslator() InTreePlugin {
	return &osCinderCSITranslator{}
}

// TranslateInTreeStorageClassToCSI translates InTree Cinder storage class parameters to CSI storage class
func (t *osCinderCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	var (
		params = map[string]string{}
	)
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case fsTypeKey:
			params[csiFsTypeKey] = v
		default:
			// All other parameters are supported by the CSI driver.
			// This includes also "availability", therefore do not translate it to sc.AllowedTopologies
			params[k] = v
		}
	}

	if len(sc.AllowedTopologies) > 0 {
		newTopologies, err := translateAllowedTopologies(sc.AllowedTopologies, CinderTopologyKey)
		if err != nil {
			return nil, fmt.Errorf("failed translating allowed topologies: %v", err)
		}
		sc.AllowedTopologies = newTopologies
	}

	sc.Parameters = params

	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with Cinder set from in-tree
// and converts the Cinder source to a CSIPersistentVolumeSource
func (t *osCinderCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.Cinder == nil {
		return nil, fmt.Errorf("volume is nil or Cinder not defined on volume")
	}

	cinderSource := volume.Cinder
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// Must be unique per disk as it is used as the unique part of the
			// staging path
			Name: fmt.Sprintf("%s-%s", CinderDriverName, cinderSource.VolumeID),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:           CinderDriverName,
					VolumeHandle:     cinderSource.VolumeID,
					ReadOnly:         cinderSource.ReadOnly,
					FSType:           cinderSource.FSType,
					VolumeAttributes: map[string]string{},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with Cinder set from in-tree
// and converts the Cinder source to a CSIPersistentVolumeSource
func (t *osCinderCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.Cinder == nil {
		return nil, fmt.Errorf("pv is nil or Cinder not defined on pv")
	}

	cinderSource := pv.Spec.Cinder

	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:           CinderDriverName,
		VolumeHandle:     cinderSource.VolumeID,
		ReadOnly:         cinderSource.ReadOnly,
		FSType:           cinderSource.FSType,
		VolumeAttributes: map[string]string{},
	}

	if err := translateTopologyFromInTreeToCSI(pv, CinderTopologyKey); err != nil {
		return nil, fmt.Errorf("failed to translate topology: %v", err)
	}

	pv.Spec.Cinder = nil
	pv.Spec.CSI = csiSource
	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the Cinder CSI source to a Cinder In-tree source.
func (t *osCinderCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}

	csiSource := pv.Spec.CSI

	cinderSource := &v1.CinderPersistentVolumeSource{
		VolumeID: csiSource.VolumeHandle,
		FSType:   csiSource.FSType,
		ReadOnly: csiSource.ReadOnly,
	}

	// translate CSI topology to In-tree topology for rollback compatibility.
	// It is not possible to guess Cinder Region from the Zone, therefore leave it empty.
	if err := translateTopologyFromCSIToInTree(pv, CinderTopologyKey, nil); err != nil {
		return nil, fmt.Errorf("failed to translate topology. PV:%+v. Error:%v", *pv, err)
	}

	pv.Spec.CSI = nil
	pv.Spec.Cinder = cinderSource
	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *osCinderCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.Cinder != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *osCinderCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.Cinder != nil
}

// GetInTreePluginName returns the name of the intree plugin driver
func (t *osCinderCSITranslator) GetInTreePluginName() string {
	return CinderInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (t *osCinderCSITranslator) GetCSIPluginName() string {
	return CinderDriverName
}

func (t *osCinderCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}
