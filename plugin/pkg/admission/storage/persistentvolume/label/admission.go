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
	"fmt"
	"io"
	"sync"

	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/features"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	vol "k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
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
	ebsVolumes       aws.Volumes
	cloudConfig      []byte
	gceCloudProvider *gce.Cloud
	azureProvider    *azure.Cloud
	vsphereProvider  *vsphere.VSphere
}

var _ admission.MutationInterface = &persistentVolumeLabel{}
var _ kubeapiserveradmission.WantsCloudConfig = &persistentVolumeLabel{}

// newPersistentVolumeLabel returns an admission.Interface implementation which adds labels to PersistentVolume CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func newPersistentVolumeLabel() *persistentVolumeLabel {
	// DEPRECATED: cloud-controller-manager will now start NewPersistentVolumeLabelController
	// which does exactly what this admission controller used to do. So once GCE, AWS and AZURE can
	// run externally, we can remove this admission controller.
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

func (l *persistentVolumeLabel) Admit(a admission.Attributes) (err error) {
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

	var volumeLabels map[string]string
	if volume.Spec.AWSElasticBlockStore != nil {
		labels, err := l.findAWSEBSLabels(volume)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("error querying AWS EBS volume %s: %v", volume.Spec.AWSElasticBlockStore.VolumeID, err))
		}
		volumeLabels = labels
	}
	if volume.Spec.GCEPersistentDisk != nil {
		labels, err := l.findGCEPDLabels(volume)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("error querying GCE PD volume %s: %v", volume.Spec.GCEPersistentDisk.PDName, err))
		}
		volumeLabels = labels
	}
	if volume.Spec.AzureDisk != nil {
		labels, err := l.findAzureDiskLabels(volume)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("error querying AzureDisk volume %s: %v", volume.Spec.AzureDisk.DiskName, err))
		}
		volumeLabels = labels
	}
	if volume.Spec.VsphereVolume != nil {
		labels, err := l.findVsphereVolumeLabels(volume)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("error querying vSphere Volume %s: %v", volume.Spec.VsphereVolume.VolumePath, err))
		}
		volumeLabels = labels
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
			if k == kubeletapis.LabelZoneFailureDomain {
				zones, err := volumeutil.LabelZonesToSet(v)
				if err != nil {
					return admission.NewForbidden(a, fmt.Errorf("failed to convert label string for Zone: %s to a Set", v))
				}
				values = zones.UnsortedList()
			} else {
				values = []string{v}
			}
			requirements = append(requirements, api.NodeSelectorRequirement{Key: k, Operator: api.NodeSelectorOpIn, Values: values})
		}

		if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
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
	}

	return nil
}

func (l *persistentVolumeLabel) findAWSEBSLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.AWSElasticBlockStore.VolumeID == vol.ProvisionedVolumeName {
		return nil, nil
	}
	ebsVolumes, err := l.getEBSVolumes()
	if err != nil {
		return nil, err
	}
	if ebsVolumes == nil {
		return nil, fmt.Errorf("unable to build AWS cloud provider for EBS")
	}

	// TODO: GetVolumeLabels is actually a method on the Volumes interface
	// If that gets standardized we can refactor to reduce code duplication
	spec := aws.KubernetesVolumeID(volume.Spec.AWSElasticBlockStore.VolumeID)
	labels, err := ebsVolumes.GetVolumeLabels(spec)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getEBSVolumes returns the AWS Volumes interface for ebs
func (l *persistentVolumeLabel) getEBSVolumes() (aws.Volumes, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.ebsVolumes == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("aws", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		awsCloudProvider, ok := cloudProvider.(*aws.Cloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving AWS cloud provider")
		}
		l.ebsVolumes = awsCloudProvider
	}
	return l.ebsVolumes, nil
}

func (l *persistentVolumeLabel) findGCEPDLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.GCEPersistentDisk.PDName == vol.ProvisionedVolumeName {
		return nil, nil
	}

	provider, err := l.getGCECloudProvider()
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return nil, fmt.Errorf("unable to build GCE cloud provider for PD")
	}

	// If the zone is already labeled, honor the hint
	zone := volume.Labels[kubeletapis.LabelZoneFailureDomain]

	labels, err := provider.GetAutoLabelsForPD(volume.Spec.GCEPersistentDisk.PDName, zone)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getGCECloudProvider returns the GCE cloud provider, for use for querying volume labels
func (l *persistentVolumeLabel) getGCECloudProvider() (*gce.Cloud, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.gceCloudProvider == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("gce", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		gceCloudProvider, ok := cloudProvider.(*gce.Cloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving GCE cloud provider")
		}
		l.gceCloudProvider = gceCloudProvider
	}
	return l.gceCloudProvider, nil
}

// getAzureCloudProvider returns the Azure cloud provider, for use for querying volume labels
func (l *persistentVolumeLabel) getAzureCloudProvider() (*azure.Cloud, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.azureProvider == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("azure", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		azureProvider, ok := cloudProvider.(*azure.Cloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving Azure cloud provider")
		}
		l.azureProvider = azureProvider
	}

	return l.azureProvider, nil
}

func (l *persistentVolumeLabel) findAzureDiskLabels(volume *api.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if volume.Spec.AzureDisk.DiskName == vol.ProvisionedVolumeName {
		return nil, nil
	}

	provider, err := l.getAzureCloudProvider()
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return nil, fmt.Errorf("unable to build Azure cloud provider for AzureDisk")
	}

	return provider.GetAzureDiskLabels(volume.Spec.AzureDisk.DataDiskURI)
}

func (l *persistentVolumeLabel) findVsphereVolumeLabels(volume *api.PersistentVolume) (map[string]string, error) {
	pvlabler, err := l.getVSphereProvider()
	if err != nil {
		return nil, err
	}
	if pvlabler == nil {
		return nil, fmt.Errorf("unable to build vSphere cloud provider")
	}

	labels, err := pvlabler.GetVolumeLabels(volume.Spec.VsphereVolume.VolumePath)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

func (l *persistentVolumeLabel) getVSphereProvider() (*vsphere.VSphere, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.vsphereProvider == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("vsphere", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		vsphereProvider, ok := cloudProvider.(*vsphere.VSphere)
		if !ok {
			// GetCloudProvider failed
			return nil, fmt.Errorf("Error retrieving vSphere Cloud Provider")
		}
		l.vsphereProvider = vsphereProvider
	}
	return l.vsphereProvider, nil
}
