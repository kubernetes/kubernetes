/*
Copyright 2015 The Kubernetes Authors.

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

package label

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	cloudprovider "k8s.io/cloud-provider"
	cloudvolume "k8s.io/cloud-provider/volume"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	persistentvolume "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	// PluginName is the name of persistent volume label admission plugin
	PluginName = "PersistentVolumeLabel"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		persistentVolumeLabelAdmission := newPersistentVolumeLabel()
		return persistentVolumeLabelAdmission, nil
	})
}

var _ = admission.Interface(&persistentVolumeLabel{})

type persistentVolumeLabel struct {
	*admission.Handler

	mutex            sync.Mutex
	cloudConfig      []byte
	awsPVLabeler     cloudprovider.PVLabeler
	gcePVLabeler     cloudprovider.PVLabeler
	azurePVLabeler   cloudprovider.PVLabeler
	vspherePVLabeler cloudprovider.PVLabeler
}

var _ admission.MutationInterface = &persistentVolumeLabel{}
var _ kubeapiserveradmission.WantsCloudConfig = &persistentVolumeLabel{}

// newPersistentVolumeLabel returns an admission.Interface implementation which adds labels to PersistentVolume CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func newPersistentVolumeLabel() *persistentVolumeLabel {
	// DEPRECATED: in a future release, we will use mutating admission webhooks to apply PV labels.
	// Once the mutating admission webhook is used for AWS, Azure and GCE,
	// this admission controller will be removed.
	klog.Warning("PersistentVolumeLabel admission controller is deprecated. " +
		"Please remove this controller from your configuration files and scripts.")
	return &persistentVolumeLabel{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (l *persistentVolumeLabel) SetCloudConfig(cloudConfig []byte) {
	l.cloudConfig = cloudConfig
}

func nodeSelectorRequirementKeysExistInNodeSelectorTerms(reqs []api.NodeSelectorRequirement, terms []api.NodeSelectorTerm) bool {
	for _, req := range reqs {
		for _, term := range terms {
			for _, r := range term.MatchExpressions {
				if r.Key == req.Key {
					return true
				}
			}
		}
	}
	return false
}

func (l *persistentVolumeLabel) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if a.GetResource().GroupResource() != api.Resource("persistentvolumes") {
		return nil
	}
	obj := a.GetObject()
	if obj == nil {
		return nil
	}
	volume, ok := obj.(*api.PersistentVolume)
	if !ok {
		return nil
	}

	volumeLabels, err := l.findVolumeLabels(volume)
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	requirements := make([]api.NodeSelectorRequirement, 0)
	if len(volumeLabels) != 0 {
		if volume.Labels == nil {
			volume.Labels = make(map[string]string)
		}
		for k, v := range volumeLabels {
			// We (silently) replace labels if they are provided.
			// This should be OK because they are in the kubernetes.io namespace
			// i.e. we own them
			volume.Labels[k] = v

			// Set NodeSelectorRequirements based on the labels
			var values []string
			if k == v1.LabelTopologyZone || k == v1.LabelFailureDomainBetaZone {
				zones, err := volumehelpers.LabelZonesToSet(v)
				if err != nil {
					return admission.NewForbidden(a, fmt.Errorf("failed to convert label string for Zone: %s to a Set", v))
				}
				// zone values here are sorted for better testability.
				values = zones.List()
			} else {
				values = []string{v}
			}
			requirements = append(requirements, api.NodeSelectorRequirement{Key: k, Operator: api.NodeSelectorOpIn, Values: values})
		}

		if volume.Spec.NodeAffinity == nil {
			volume.Spec.NodeAffinity = new(api.VolumeNodeAffinity)
		}
		if volume.Spec.NodeAffinity.Required == nil {
			volume.Spec.NodeAffinity.Required = new(api.NodeSelector)
		}
		if len(volume.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
			// Need at least one term pre-allocated whose MatchExpressions can be appended to
			volume.Spec.NodeAffinity.Required.NodeSelectorTerms = make([]api.NodeSelectorTerm, 1)
		}
		if nodeSelectorRequirementKeysExistInNodeSelectorTerms(requirements, volume.Spec.NodeAffinity.Required.NodeSelectorTerms) {
			klog.V(4).Infof("NodeSelectorRequirements for cloud labels %v conflict with existing NodeAffinity %v. Skipping addition of NodeSelectorRequirements for cloud labels.",
				requirements, volume.Spec.NodeAffinity)
		} else {
			for _, req := range requirements {
				for i := range volume.Spec.NodeAffinity.Required.NodeSelectorTerms {
					volume.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions = append(volume.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions, req)
				}
			}
		}
	}

	return nil
}

func (l *persistentVolumeLabel) findVolumeLabels(volume *api.PersistentVolume) (map[string]string, error) {
	existingLabels := volume.Labels

	// All cloud providers set only these two labels.
	topologyLabelGA := true
	domain, domainOK := existingLabels[v1.LabelTopologyZone]
	region, regionOK := existingLabels[v1.LabelTopologyRegion]
	// If they don't have GA labels we should check for failuredomain beta labels
	// TODO: remove this once all the cloud provider change to GA topology labels
	if !domainOK || !regionOK {
		topologyLabelGA = false
		domain, domainOK = existingLabels[v1.LabelFailureDomainBetaZone]
		region, regionOK = existingLabels[v1.LabelFailureDomainBetaRegion]
	}

	isDynamicallyProvisioned := metav1.HasAnnotation(volume.ObjectMeta, persistentvolume.AnnDynamicallyProvisioned)
	if isDynamicallyProvisioned && domainOK && regionOK {
		// PV already has all the labels and we can trust the dynamic provisioning that it provided correct values.
		if topologyLabelGA {
			return map[string]string{
				v1.LabelTopologyZone:   domain,
				v1.LabelTopologyRegion: region,
			}, nil
		}
		return map[string]string{
			v1.LabelFailureDomainBetaZone:   domain,
			v1.LabelFailureDomainBetaRegion: region,
		}, nil

	}

	// Either missing labels or we don't trust the user provided correct values.
	switch {
	case volume.Spec.AWSElasticBlockStore != nil:
		labels, err := l.findAWSEBSLabels(volume)
		if err != nil {
			return nil, fmt.Errorf("error querying AWS EBS volume %s: %v", volume.Spec.AWSElasticBlockStore.VolumeID, err)
		}
		return labels, nil
	case volume.Spec.GCEPersistentDisk != nil:
		labels, err := l.findGCEPDLabels(volume)
		if err != nil {
			return nil, fmt.Errorf("error querying GCE PD volume %s: %v", volume.Spec.GCEPersistentDisk.PDName, err)
		}
		return labels, nil
	case volume.Spec.AzureDisk != nil:
		labels, err := l.findAzureDiskLabels(volume)
		if err != nil {
			return nil, fmt.Errorf("error querying AzureDisk volume %s: %v", volume.Spec.AzureDisk.DiskName, err)
		}
		return labels, nil
	case volume.Spec.VsphereVolume != nil:
		labels, err := l.findVsphereVolumeLabels(volume)
		if err != nil {
			return nil, fmt.Errorf("error querying vSphere Volume %s: %v", volume.Spec.VsphereVolume.VolumePath, err)
		}
		return labels, nil
	}
	// Unrecognized volume, do not add any labels
	return nil, nil
}

func (l *persistentVolumeLabel) findAWSEBSLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.AWSElasticBlockStore.VolumeID == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}
	pvlabler, err := l.getAWSPVLabeler()
	if err != nil {
		return nil, err
	}
	if pvlabler == nil {
		return nil, fmt.Errorf("unable to build AWS cloud provider for EBS")
	}

	pv := &v1.PersistentVolume{}
	err = k8s_api_v1.Convert_core_PersistentVolume_To_v1_PersistentVolume(volume, pv, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert PersistentVolume to core/v1: %q", err)
	}

	return pvlabler.GetLabelsForVolume(context.TODO(), pv)
}

// getAWSPVLabeler returns the AWS implementation of PVLabeler
func (l *persistentVolumeLabel) getAWSPVLabeler() (cloudprovider.PVLabeler, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.awsPVLabeler == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}

		cloudProvider, err := cloudprovider.GetCloudProvider("aws", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}

		awsPVLabeler, ok := cloudProvider.(cloudprovider.PVLabeler)
		if !ok {
			return nil, errors.New("AWS cloud provider does not implement PV labeling")
		}

		l.awsPVLabeler = awsPVLabeler
	}
	return l.awsPVLabeler, nil
}

func (l *persistentVolumeLabel) findGCEPDLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.GCEPersistentDisk.PDName == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}

	pvlabler, err := l.getGCEPVLabeler()
	if err != nil {
		return nil, err
	}
	if pvlabler == nil {
		return nil, fmt.Errorf("unable to build GCE cloud provider for PD")
	}

	pv := &v1.PersistentVolume{}
	err = k8s_api_v1.Convert_core_PersistentVolume_To_v1_PersistentVolume(volume, pv, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert PersistentVolume to core/v1: %q", err)
	}
	return pvlabler.GetLabelsForVolume(context.TODO(), pv)
}

// getGCEPVLabeler returns the GCE implementation of PVLabeler
func (l *persistentVolumeLabel) getGCEPVLabeler() (cloudprovider.PVLabeler, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.gcePVLabeler == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}

		cloudProvider, err := cloudprovider.GetCloudProvider("gce", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}

		gcePVLabeler, ok := cloudProvider.(cloudprovider.PVLabeler)
		if !ok {
			return nil, errors.New("GCE cloud provider does not implement PV labeling")
		}

		l.gcePVLabeler = gcePVLabeler

	}
	return l.gcePVLabeler, nil
}

// getAzurePVLabeler returns the Azure implementation of PVLabeler
func (l *persistentVolumeLabel) getAzurePVLabeler() (cloudprovider.PVLabeler, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.azurePVLabeler == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}

		cloudProvider, err := cloudprovider.GetCloudProvider("azure", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}

		azurePVLabeler, ok := cloudProvider.(cloudprovider.PVLabeler)
		if !ok {
			return nil, errors.New("Azure cloud provider does not implement PV labeling")
		}
		l.azurePVLabeler = azurePVLabeler
	}

	return l.azurePVLabeler, nil
}

func (l *persistentVolumeLabel) findAzureDiskLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.AzureDisk.DiskName == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}

	pvlabler, err := l.getAzurePVLabeler()
	if err != nil {
		return nil, err
	}
	if pvlabler == nil {
		return nil, fmt.Errorf("unable to build Azure cloud provider for AzureDisk")
	}

	pv := &v1.PersistentVolume{}
	err = k8s_api_v1.Convert_core_PersistentVolume_To_v1_PersistentVolume(volume, pv, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert PersistentVolume to core/v1: %q", err)
	}
	return pvlabler.GetLabelsForVolume(context.TODO(), pv)
}

func (l *persistentVolumeLabel) findVsphereVolumeLabels(volume *api.PersistentVolume) (map[string]string, error) {
	pvlabler, err := l.getVspherePVLabeler()
	if err != nil {
		return nil, err
	}
	if pvlabler == nil {
		return nil, fmt.Errorf("unable to build vSphere cloud provider")
	}

	pv := &v1.PersistentVolume{}
	err = k8s_api_v1.Convert_core_PersistentVolume_To_v1_PersistentVolume(volume, pv, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert PersistentVolume to core/v1: %q", err)
	}
	labels, err := pvlabler.GetLabelsForVolume(context.TODO(), pv)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

func (l *persistentVolumeLabel) getVspherePVLabeler() (cloudprovider.PVLabeler, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.vspherePVLabeler == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("vsphere", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		vspherePVLabeler, ok := cloudProvider.(cloudprovider.PVLabeler)
		if !ok {
			// GetCloudProvider failed
			return nil, errors.New("vSphere Cloud Provider does not implement PV labeling")
		}
		l.vspherePVLabeler = vspherePVLabeler
	}
	return l.vspherePVLabeler, nil
}
