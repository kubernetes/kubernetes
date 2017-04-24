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

package vSphereSPBMPolicy

import (
	"bytes"
	"fmt"
	"io"
	"sync"

	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/admission"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

func init() {
	admission.RegisterPlugin("VSphereSPBMPolicyManager", func(config io.Reader) (admission.Interface, error) {
		vSphereSPBMPolicyManager := NewVSphereSPBMPolicyManager()
		return vSphereSPBMPolicyManager, nil
	})
}

var _ = admission.Interface(&vSphereSPBMPolicyManager{})

type vSphereSPBMPolicyManager struct {
	*admission.Handler

	mutex                sync.Mutex
	cloudConfig          []byte
	vSphereCloudProvider *vsphere.VSphere
}

var _ kubeapiserveradmission.WantsCloudConfig = &vSphereSPBMPolicyManager{}

// NewVSphereSPBMPolicyManager returns an admission.Interface implementation which create SPBM storage policies to StorageClass CREATE requests.
func NewVSphereSPBMPolicyManager() *vSphereSPBMPolicyManager {
	return &vSphereSPBMPolicyManager{
		// This handler gets executed only for CREATE/DELETE operation.
		Handler: admission.NewHandler(admission.Create, admission.Delete, admission.Update),
	}
}

func (l *vSphereSPBMPolicyManager) SetCloudConfig(cloudConfig []byte) {
	l.cloudConfig = cloudConfig
}

func (l *vSphereSPBMPolicyManager) Admit(a admission.Attributes) (err error) {
	glog.V(1).Infof("balu - In vsphere admission controller Admit() method, group resource is %+v", a.GetResource().GroupResource())
	glog.V(1).Infof("balu - In vsphere admission controller Admit() method above all, operation is %+q", a.GetOperation())
	// Make sure this request is only handled for storage classes API objects.
	// For all other requests, just return.
	if a.GetResource().GroupResource() != storageapi.Resource("storageclasses") {
		return nil
	}

	// Get the storage class API object.
	obj := a.GetObject()
	if obj == nil {
		glog.V(1).Infof("balu - In vsphere admission controller Admit(), obj is null")
		return nil
	}
	storageClass, ok := obj.(*storageapi.StorageClass)
	if !ok {
		glog.V(1).Infof("balu - In vsphere admission controller Admit(), obj: %+v is not storage class", obj)
		return nil
	}

	var storageClassParams map[string]string
	glog.V(1).Infof("balu - In vsphere admission controller Admit(), storage class provisioner is: %+q", storageClass.Provisioner)
	// Check if provisioner is not empty and execute only if it is a vsphere volume provisioner.
	// For other provisioners just return back.
	if storageClass.Provisioner != "" && storageClass.Provisioner == "kubernetes.io/vsphere-volume" {
		// Get the storage class parameters.
		storageClassParams = storageClass.Parameters
		glog.V(1).Infof("balu - In vsphere admission controller, inside storageclass paramaters %+q", storageClassParams)
		// Check for vmName parameter. Continue if exists, otherwise return.
		if storageClassParams["vmName"] == "" {
			glog.V(1).Infof("balu - In vsphere admission controller, no vmName paramater, operation: %q", a.GetOperation())
			return nil
		}

		// Get the vSphere cloud provider instance registered with the API server.
		provider, err := l.getVSphereCloudProvider()
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("balu - unable to get vSphere cloud provider with err: %+v", err))
		}
		if provider == nil {
			return admission.NewForbidden(a, fmt.Errorf("balu - unable to get vSphere cloud provider"))
		}

		glog.V(1).Infof("balu - In vsphere admission controller, operation is %+q", a.GetOperation())

		// Check if the operation is STORAGE CLASS CREATE or DELETE.
		if a.GetOperation() == admission.Create {
			glog.V(1).Infof("balu - In vsphere admission controller, inside create operation is %+q", a.GetOperation())
			// In case of CREATE STORAGE CLASS, we create a VM specified in the YAML.
			err = provider.CreateVMWithAdmissionPlugin(storageClassParams["vmName"])
			if err != nil {
				return admission.NewForbidden(a, fmt.Errorf("balu - unable to create a VM with these parameters : %+q", storageClassParams["vmName"]))
			}
		} else if a.GetOperation() == admission.Delete {
			glog.V(1).Infof("balu - In vsphere admission controller, inside delete operation is %+q", a.GetOperation())
			// In case of DELETE STORAGE CLASS, we delete a VM specified in the YAML.
			err = provider.DeleteVMWithAdmissionPlugin(storageClassParams["vmName"])
			if err != nil {
				return admission.NewForbidden(a, fmt.Errorf("balu - unable to delete a VM with these parameters : %+q", storageClassParams["vmName"]))
			}
		}
	}

	// Success
	return nil
}

// getvSphereCloudProvider returns the vSphere cloud provider
func (l *vSphereSPBMPolicyManager) getVSphereCloudProvider() (*vsphere.VSphere, error) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.vSphereCloudProvider == nil {
		var cloudConfigReader io.Reader
		if len(l.cloudConfig) > 0 {
			cloudConfigReader = bytes.NewReader(l.cloudConfig)
		}
		cloudProvider, err := cloudprovider.GetCloudProvider("vsphere", cloudConfigReader)
		if err != nil || cloudProvider == nil {
			return nil, err
		}
		vSphereCloudProvider, ok := cloudProvider.(*vsphere.VSphere)
		if !ok {
			// GetCloudProvider has gone very wrong
			return nil, fmt.Errorf("error retrieving vSphere cloud provider")
		}
		l.vSphereCloudProvider = vSphereCloudProvider
	}
	return l.vSphereCloudProvider, nil
}
