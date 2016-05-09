/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"github.com/golang/glog"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/volume"
)

func init() {
	admission.RegisterPlugin("PersistentVolumeLabel", func(client clientset.Interface, config io.Reader, host admission.AdmissionPluginHost) (admission.Interface, error) {
		persistentVolumeLabelAdmission := NewPersistentVolumeLabel(host)
		return persistentVolumeLabelAdmission, nil
	})
}

var _ = admission.Interface(&persistentVolumeLabel{})

type persistentVolumeLabel struct {
	*admission.Handler

	mutex            sync.Mutex
	ebsVolumes       aws.Volumes
	gceCloudProvider *gce.GCECloud
	host             admission.AdmissionPluginHost
}

// NewPersistentVolumeLabel returns an admission.Interface implementation which adds labels to PersistentVolume CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func NewPersistentVolumeLabel(host admission.AdmissionPluginHost) *persistentVolumeLabel {
	return &persistentVolumeLabel{
		Handler: admission.NewHandler(admission.Create),
		host:    host,
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
	vol, ok := obj.(*api.PersistentVolume)
	if !ok {
		return nil
	}

	volSpec := &volume.Spec{
		PersistentVolume: vol,
	}
	volumePluginMgr := l.host.GetVolumePluginMgr()
	volumePlugin, err := volumePluginMgr.FindPluginBySpec(volSpec)
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	labelerPlugin, ok := volumePlugin.(volume.VolumeLabelerPlugin)
	if !ok {
		// the volume plugin does not implement VolumeLabeler interface and thus
		// does not support labels -> allow
		glog.V(5).Infof("Ignoring PersistentVolume %s: plugin does not implement VolumeLabeler interface", vol.ObjectMeta.Name)
		return nil
	}

	labeler, err := labelerPlugin.NewVolumeLabeler(volSpec)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error creating labeler for volume: %v", err))
	}

	labels, err := labeler.GetLabels()
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error getting labels for volume: %v", err))
	}

	if len(labels) != 0 {
		if vol.Labels == nil {
			vol.Labels = make(map[string]string)

		}
		for k, v := range labels {
			// We (silently) replace labels if they are provided.
			// This should be OK because they are in the kubernetes.io namespace
			// i.e. we own them
			vol.Labels[k] = v
		}
	}

	return nil
}
