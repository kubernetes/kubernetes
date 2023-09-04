/*
Copyright 2023 The Kubernetes Authors.

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

package setdefault

import (
	"context"
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	storagev1listers "k8s.io/client-go/listers/storage/v1"
	storagev1alpha1listers "k8s.io/client-go/listers/storage/v1alpha1"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	// PluginName is the name of this admission controller plugin
	PluginName = "DefaultVolumeAttributesClass"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin()
		return plugin, nil
	})
}

// claimDefaulterPlugin holds state for and implements the admission plugin.
type claimDefaulterPlugin struct {
	*admission.Handler

	lister   storagev1alpha1listers.VolumeAttributesClassLister
	sclister storagev1listers.StorageClassLister
}

var _ admission.Interface = &claimDefaulterPlugin{}
var _ admission.MutationInterface = &claimDefaulterPlugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&claimDefaulterPlugin{})

// newPlugin creates a new admission plugin.
func newPlugin() *claimDefaulterPlugin {
	return &claimDefaulterPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (a *claimDefaulterPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	informer := f.Storage().V1alpha1().VolumeAttributesClasses()
	a.lister = informer.Lister()
	scinformer := f.Storage().V1().StorageClasses()
	a.sclister = scinformer.Lister()
	a.SetReadyFunc(func() bool {
		return informer.Informer().HasSynced() && scinformer.Informer().HasSynced()
	})
}

// ValidateInitialization ensures lister is set.
func (a *claimDefaulterPlugin) ValidateInitialization() error {
	if a.lister == nil {
		return fmt.Errorf("missing lister")
	}
	if a.sclister == nil {
		return fmt.Errorf("missing storage class lister")
	}
	return nil
}

// Admit sets the default value of a PersistentVolumeClaim's volume attributes class, in case the user did
// not provide a value.
//
// 1.  Find available VolumeAttributesClass.
// 2.  Figure which is the default
// 3.  Write to the PVClaim
func (a *claimDefaulterPlugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != api.Resource("persistentvolumeclaims") {
		return nil
	}

	if len(attr.GetSubresource()) != 0 {
		return nil
	}

	pvc, ok := attr.GetObject().(*api.PersistentVolumeClaim)
	// if we can't convert then we don't handle this object so just return
	if !ok {
		return nil
	}

	if pvc.Spec.VolumeAttributesClassName != nil {
		// The user asked for a volume attributes class.
		return nil
	}

	klog.V(4).Infof("no volume attributes class for claim %s (generate: %s)", pvc.Name, pvc.GenerateName)

	if pvc.Spec.StorageClassName == nil || *pvc.Spec.StorageClassName == "" {
		return nil
	}

	sc, err := a.sclister.Get(*pvc.Spec.StorageClassName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// The storage class does not exist, do nothing about the PVC.
			return nil
		}
		return admission.NewForbidden(attr, err)
	}

	def, err := util.GetDefaultVolumeAttributesClass(a.lister, sc.Provisioner)
	if err != nil {
		return admission.NewForbidden(attr, err)
	}
	if def == nil {
		// No default class selected, do nothing about the PVC.
		return nil
	}

	klog.V(4).Infof("defaulting volume attributes class for claim %s (generate: %s) to %s", pvc.Name, pvc.GenerateName, def.Name)
	pvc.Spec.VolumeAttributesClassName = &def.Name
	return nil
}
