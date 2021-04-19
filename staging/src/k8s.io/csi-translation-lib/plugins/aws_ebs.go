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
	"net/url"
	"regexp"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// AWSEBSDriverName is the name of the CSI driver for EBS
	AWSEBSDriverName = "ebs.csi.aws.com"
	// AWSEBSInTreePluginName is the name of the intree plugin for EBS
	AWSEBSInTreePluginName = "kubernetes.io/aws-ebs"
	// AWSEBSTopologyKey is the zonal topology key for AWS EBS CSI driver
	AWSEBSTopologyKey = "topology." + AWSEBSDriverName + "/zone"
)

var _ InTreePlugin = &awsElasticBlockStoreCSITranslator{}

// awsElasticBlockStoreTranslator handles translation of PV spec from In-tree EBS to CSI EBS and vice versa
type awsElasticBlockStoreCSITranslator struct{}

// NewAWSElasticBlockStoreCSITranslator returns a new instance of awsElasticBlockStoreTranslator
func NewAWSElasticBlockStoreCSITranslator() InTreePlugin {
	return &awsElasticBlockStoreCSITranslator{}
}

// TranslateInTreeStorageClassToCSI translates InTree EBS storage class parameters to CSI storage class
func (t *awsElasticBlockStoreCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	var (
		generatedTopologies []v1.TopologySelectorTerm
		params              = map[string]string{}
	)
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case fsTypeKey:
			params[csiFsTypeKey] = v
		case zoneKey:
			generatedTopologies = generateToplogySelectors(AWSEBSTopologyKey, []string{v})
		case zonesKey:
			generatedTopologies = generateToplogySelectors(AWSEBSTopologyKey, strings.Split(v, ","))
		default:
			params[k] = v
		}
	}

	if len(generatedTopologies) > 0 && len(sc.AllowedTopologies) > 0 {
		return nil, fmt.Errorf("cannot simultaneously set allowed topologies and zone/zones parameters")
	} else if len(generatedTopologies) > 0 {
		sc.AllowedTopologies = generatedTopologies
	} else if len(sc.AllowedTopologies) > 0 {
		newTopologies, err := translateAllowedTopologies(sc.AllowedTopologies, AWSEBSTopologyKey)
		if err != nil {
			return nil, fmt.Errorf("failed translating allowed topologies: %v", err)
		}
		sc.AllowedTopologies = newTopologies
	}

	sc.Parameters = params

	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with AWSElasticBlockStore set from in-tree
// and converts the AWSElasticBlockStore source to a CSIPersistentVolumeSource
func (t *awsElasticBlockStoreCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.AWSElasticBlockStore == nil {
		return nil, fmt.Errorf("volume is nil or AWS EBS not defined on volume")
	}
	ebsSource := volume.AWSElasticBlockStore
	volumeHandle, err := KubernetesVolumeIDToEBSVolumeID(ebsSource.VolumeID)
	if err != nil {
		return nil, fmt.Errorf("failed to translate Kubernetes ID to EBS Volume ID %v", err)
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// Must be unique per disk as it is used as the unique part of the
			// staging path
			Name: fmt.Sprintf("%s-%s", AWSEBSDriverName, volumeHandle),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       AWSEBSDriverName,
					VolumeHandle: volumeHandle,
					ReadOnly:     ebsSource.ReadOnly,
					FSType:       ebsSource.FSType,
					VolumeAttributes: map[string]string{
						"partition": strconv.FormatInt(int64(ebsSource.Partition), 10),
					},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with AWSElasticBlockStore set from in-tree
// and converts the AWSElasticBlockStore source to a CSIPersistentVolumeSource
func (t *awsElasticBlockStoreCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.AWSElasticBlockStore == nil {
		return nil, fmt.Errorf("pv is nil or AWS EBS not defined on pv")
	}

	ebsSource := pv.Spec.AWSElasticBlockStore

	volumeHandle, err := KubernetesVolumeIDToEBSVolumeID(ebsSource.VolumeID)
	if err != nil {
		return nil, fmt.Errorf("failed to translate Kubernetes ID to EBS Volume ID %v", err)
	}

	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:       AWSEBSDriverName,
		VolumeHandle: volumeHandle,
		ReadOnly:     ebsSource.ReadOnly,
		FSType:       ebsSource.FSType,
		VolumeAttributes: map[string]string{
			"partition": strconv.FormatInt(int64(ebsSource.Partition), 10),
		},
	}

	if err := translateTopology(pv, AWSEBSTopologyKey); err != nil {
		return nil, fmt.Errorf("failed to translate topology: %v", err)
	}

	pv.Spec.AWSElasticBlockStore = nil
	pv.Spec.CSI = csiSource
	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the EBS CSI source to a AWSElasticBlockStore source.
func (t *awsElasticBlockStoreCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}

	csiSource := pv.Spec.CSI

	ebsSource := &v1.AWSElasticBlockStoreVolumeSource{
		VolumeID: csiSource.VolumeHandle,
		FSType:   csiSource.FSType,
		ReadOnly: csiSource.ReadOnly,
	}

	if partition, ok := csiSource.VolumeAttributes["partition"]; ok {
		partValue, err := strconv.Atoi(partition)
		if err != nil {
			return nil, fmt.Errorf("failed to convert partition %v to integer: %v", partition, err)
		}
		ebsSource.Partition = int32(partValue)
	}

	pv.Spec.CSI = nil
	pv.Spec.AWSElasticBlockStore = ebsSource
	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *awsElasticBlockStoreCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.AWSElasticBlockStore != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *awsElasticBlockStoreCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.AWSElasticBlockStore != nil
}

// GetInTreePluginName returns the name of the intree plugin driver
func (t *awsElasticBlockStoreCSITranslator) GetInTreePluginName() string {
	return AWSEBSInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (t *awsElasticBlockStoreCSITranslator) GetCSIPluginName() string {
	return AWSEBSDriverName
}

func (t *awsElasticBlockStoreCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}

// awsVolumeRegMatch represents Regex Match for AWS volume.
var awsVolumeRegMatch = regexp.MustCompile("^vol-[^/]*$")

// KubernetesVolumeIDToEBSVolumeID translates Kubernetes volume ID to EBS volume ID
// KubernetsVolumeID forms:
//  * aws://<zone>/<awsVolumeId>
//  * aws:///<awsVolumeId>
//  * <awsVolumeId>
// EBS Volume ID form:
//  * vol-<alphanumberic>
// This translation shouldn't be needed and should be fixed in long run
// See https://github.com/kubernetes/kubernetes/issues/73730
func KubernetesVolumeIDToEBSVolumeID(kubernetesID string) (string, error) {
	// name looks like aws://availability-zone/awsVolumeId

	// The original idea of the URL-style name was to put the AZ into the
	// host, so we could find the AZ immediately from the name without
	// querying the API.  But it turns out we don't actually need it for
	// multi-AZ clusters, as we put the AZ into the labels on the PV instead.
	// However, if in future we want to support multi-AZ cluster
	// volume-awareness without using PersistentVolumes, we likely will
	// want the AZ in the host.
	if !strings.HasPrefix(kubernetesID, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		return kubernetesID, nil
	}
	url, err := url.Parse(kubernetesID)
	if err != nil {
		// TODO: Maybe we should pass a URL into the Volume functions
		return "", fmt.Errorf("Invalid disk name (%s): %v", kubernetesID, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", kubernetesID)
	}

	awsID := url.Path
	awsID = strings.Trim(awsID, "/")

	// We sanity check the resulting volume; the two known formats are
	// vol-12345678 and vol-12345678abcdef01
	if !awsVolumeRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", kubernetesID)
	}

	return awsID, nil
}
