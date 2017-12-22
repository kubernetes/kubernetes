/*
Copyright 2017 The Kubernetes Authors.

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

package pvcprotection

import (
	"fmt"
	"io"

	"github.com/golang/glog"

	admission "k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/features"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// PluginName is the name of this admission controller plugin
	PluginName = "PVCProtection"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin()
		return plugin, nil
	})
}

// pvcProtectionPlugin holds state for and implements the admission plugin.
type pvcProtectionPlugin struct {
	*admission.Handler
	lister corelisters.PersistentVolumeClaimLister
}

var _ admission.Interface = &pvcProtectionPlugin{}
var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&pvcProtectionPlugin{})

// newPlugin creates a new admission plugin.
func newPlugin() *pvcProtectionPlugin {
	return &pvcProtectionPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (c *pvcProtectionPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	informer := f.Core().InternalVersion().PersistentVolumeClaims()
	c.lister = informer.Lister()
	c.SetReadyFunc(informer.Informer().HasSynced)
}

// ValidateInitialization ensures lister is set.
func (c *pvcProtectionPlugin) ValidateInitialization() error {
	if c.lister == nil {
		return fmt.Errorf("missing lister")
	}
	return nil
}

// Admit sets finalizer on all PVCs. The finalizer is removed by
// PVCProtectionController when it's not referenced by any pod.
//
// This prevents users from deleting a PVC that's used by a running pod.
func (c *pvcProtectionPlugin) Admit(a admission.Attributes) error {
	if !feature.DefaultFeatureGate.Enabled(features.PVCProtection) {
		return nil
	}

	if a.GetResource().GroupResource() != api.Resource("persistentvolumeclaims") {
		return nil
	}

	if len(a.GetSubresource()) != 0 {
		return nil
	}

	pvc, ok := a.GetObject().(*api.PersistentVolumeClaim)
	// if we can't convert then we don't handle this object so just return
	if !ok {
		return nil
	}

	for _, f := range pvc.Finalizers {
		if f == volumeutil.PVCProtectionFinalizer {
			// Finalizer is already present, nothing to do
			return nil
		}
	}

	glog.V(4).Infof("adding PVC protection finalizer to %s/%s", pvc.Namespace, pvc.Name)
	pvc.Finalizers = append(pvc.Finalizers, volumeutil.PVCProtectionFinalizer)
	return nil
}
