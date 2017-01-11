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
	"fmt"
	"io"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	vol "k8s.io/kubernetes/pkg/volume"
)

func init() {
	admission.RegisterPlugin("PersistentVolumeLabel", func(config io.Reader) (admission.Interface, error) {
		persistentVolumeLabelAdmission := NewPersistentVolumeLabel()
		return persistentVolumeLabelAdmission, nil
	})
}

var _ = admission.Interface(&persistentVolumeLabel{})

type persistentVolumeLabel struct {
	*admission.Handler

	mutex            sync.Mutex
	ebsVolumes       aws.Volumes
	gceCloudProvider *gce.GCECloud
}

// NewPersistentVolumeLabel returns an admission.Interface implementation which adds labels to PersistentVolume CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func NewPersistentVolumeLabel() *persistentVolumeLabel {
	return &persistentVolumeLabel{
		Handler: admission.NewHandler(admission.Create),
	}
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

	if len(volumeLabels) != 0 {
		if volume.Labels == nil {
			volume.Labels = make(map[string]string)
		}
		for k, v := range volumeLabels {
			// We (silently) replace labels if they are provided.
			// This should be OK because they are in the kubernetes.io namespace
			// i.e. we own them
			volume.Labels[k] = v
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
		cloudProvider, err := cloudprovider.GetCloudProvider("aws", nil)
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
	zone := volume.Labels[metav1.LabelZoneFailureDomain]

	labels, err := provider.GetAutoLabelsForPD(volume.Spec.GCEPersistentDisk.PDName, zone)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getGCECloudProvider returns the GCE cloud provider, for use for querying volume labels
func (l *persistentVolumeLabel) getGCECloudProvider() (*gce.GCECloud, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.gceCloudProvider == nil {
		cloudProvider, err := cloudprovider.GetCloudProvider("gce", nil)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		gceCloudProvider, ok := cloudProvider.(*gce.GCECloud)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving GCE cloud provider")
		}
		l.gceCloudProvider = gceCloudProvider
	}
	return l.gceCloudProvider, nil
}
