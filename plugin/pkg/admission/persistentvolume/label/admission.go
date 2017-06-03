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

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	vol "k8s.io/kubernetes/pkg/volume"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("PersistentVolumeLabel", func(config io.Reader) (admission.Interface, error) {
		persistentVolumeLabelAdmission := NewPersistentVolumeLabel()
		return persistentVolumeLabelAdmission, nil
	})
}

var _ = admission.Interface(&persistentVolumeLabel{})

type persistentVolumeLabel struct {
	*admission.Handler
	client           internalclientset.Interface
	podLister        corelisters.PodLister
	mutex            sync.Mutex
	ebsVolumes       aws.Volumes
	cloudConfig      []byte
	gceCloudProvider *gce.GCECloud
}

var _ kubeapiserveradmission.WantsCloudConfig = &persistentVolumeLabel{}

// NewPersistentVolumeLabel returns an admission.Interface implementation which adds labels to PersistentVolume CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func NewPersistentVolumeLabel() *persistentVolumeLabel {
	return &persistentVolumeLabel{
		Handler: admission.NewHandler(admission.Create, admission.Delete),
	}
}

func (l *persistentVolumeLabel) SetCloudConfig(cloudConfig []byte) {
	l.cloudConfig = cloudConfig
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&persistentVolumeLabel{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&persistentVolumeLabel{})

func (l *persistentVolumeLabel) SetInternalKubeClientSet(client internalclientset.Interface) {
	l.client = client
}

func (l *persistentVolumeLabel) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	podInformer := f.Core().InternalVersion().Pods()
	l.SetReadyFunc(podInformer.Informer().HasSynced)
	l.podLister = podInformer.Lister()
}

func (l *persistentVolumeLabel) Validate() error {
	if l.podLister == nil {
		return fmt.Errorf("missing podLister")
	}
	if l.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

func (l *persistentVolumeLabel) Admit(a admission.Attributes) (err error) {
	glog.V(1).Infof("IDC - Admit")
	if a.GetOperation() == admission.Delete && a.GetKind().GroupKind() == api.Kind("PersistentVolumeClaim") {
		glog.V(1).Infof("IDC - pvc delete!")
		glog.V(1).Infof("IDC - a.GetName() type %s", a.GetName())

		pvc, pvcErr := l.client.Core().PersistentVolumeClaims(a.GetNamespace()).Get(a.GetName(), metav1.GetOptions{})
		// if we can't convert then we don't handle this object so just return
		if pvcErr != nil {
			glog.V(1).Infof("IDC - pvcErr")
			//TODO log error since we didnt find this object, but allow delete to proceed
			return nil
		}

		glog.V(1).Infof("IDC - pvc.Name %s", pvc.Name)
		glog.V(1).Infof("IDC - pvc.Namespace %s", pvc.Namespace)

		val := l.isPVCOkToDelete(pvc)
		if val == false {
			glog.V(1).Infof("IDC - l.isPVCOkToDelete == false")
			return admission.NewForbidden(a, fmt.Errorf("IDC - pvc %s is referenced by an active pod", pvc.Name))
		}
		glog.V(1).Infof("IDC - l.isPVCOkToDelete == true")
		return nil
	}
	if a.GetOperation() != admission.Create && a.GetResource().GroupResource() != api.Resource("persistentvolumes") {
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

func (l *persistentVolumeLabel) isPVCOkToDelete(pvc *api.PersistentVolumeClaim) bool {
	// prevent deletion of a pvc that is bound to a pv
	// if the pv is used by an active pod
	// issue #45143

	glog.Infof("IDC isPVCOkToDelete entered")

	//check if pvc is bound to pv and associated pv name exists
	if pvc != nil && pvc.Status.Phase == api.ClaimBound && pvc.Spec.VolumeName != "" {
		glog.Infof("IDC isPVCOkToDelete pvc is bound")
		//find the associated pv and check
		volume, volumeErr := l.client.Core().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
		if volumeErr == nil {
			glog.Infof("IDC isPVCOkToDelete found bound volume %s", volume.Name)
			//check all running and pending pods for this volume
			pods, podErr := l.podLister.Pods(pvc.Namespace).List(labels.Everything())
			if podErr == nil {
				glog.Infof("IDC isPVCOkToDelete got pods")
				for _, pod := range pods {
					glog.Infof("IDC isPVCOkToDelete checking pod %q", pod.Name)
					if pod.Status.Phase == api.PodRunning || pod.Status.Phase == api.PodPending {
						glog.Infof("IDC isPVCOkToDelete pod %q running or pending", pod.Name)
						for _, podVolume := range pod.Spec.Volumes {
							glog.Infof("IDC isPVCOkToDelete pod check volume %s", podVolume.Name)
							if podVolume.VolumeSource.PersistentVolumeClaim != nil {
								glog.Infof("IDC isPVCOkToDelete pod check volume %s is pvc", podVolume.Name)
								glog.Infof("IDC isPVCOkToDelete podVolume.VolumeSource.PersistentVolumeClaim.ClaimName is <%s>", podVolume.VolumeSource.PersistentVolumeClaim.ClaimName)
								glog.Infof("IDC isPVCOkToDelete pvc.Spec.VolumeName is <%s>", pvc.Spec.VolumeName)
								glog.Infof("IDC isPVCOkToDelete pvc.Name is <%s>", pvc.Name)
								if podVolume.VolumeSource.PersistentVolumeClaim.ClaimName == pvc.Name {
									glog.Infof("IDC isPVCOkToDelete found matching active volume")
									//found that this pvc uses a pv that is used by an running or pending pod
									glog.Infof("pvc %q was not deleted because it is associated with pv %q used by active pod %q:%q", pvc.Name, pvc.Spec.VolumeName, volume.Name, pod.Namespace, pod.Name)
									return false
								}
							}
						}
					}
				}
			}
		}
	}
	return true
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
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("gce", cloudConfigReader)
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
