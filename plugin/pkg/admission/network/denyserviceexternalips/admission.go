/*
Copyright 2020 The Kubernetes Authors.

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

package denyserviceexternalips

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	// PluginName is the name of this admission controller plugin
	PluginName = "DenyServiceExternalIPs"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin()
		return plugin, nil
	})
}

// externalIPsDenierPlugin holds state for and implements the admission plugin.
type externalIPsDenierPlugin struct {
	*admission.Handler
}

var _ admission.Interface = &externalIPsDenierPlugin{}
var _ admission.ValidationInterface = &externalIPsDenierPlugin{}

// newPlugin creates a new admission plugin.
func newPlugin() *externalIPsDenierPlugin {
	return &externalIPsDenierPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Admit ensures that modifications of the Service.Spec.ExternalIPs field are
// denied
func (plug *externalIPsDenierPlugin) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != core.Resource("services") {
		return nil
	}

	if len(attr.GetSubresource()) != 0 {
		return nil
	}

	// if we can't convert then we don't handle this object so just return
	newSvc, ok := attr.GetObject().(*core.Service)
	if !ok {
		klog.V(3).Infof("Expected Service resource, got: %v", attr.GetKind())
		return errors.NewInternalError(fmt.Errorf("Expected Service resource, got: %v", attr.GetKind()))
	}

	var oldSvc *core.Service
	if old := attr.GetOldObject(); old != nil {
		tmp, ok := old.(*core.Service)
		if !ok {
			klog.V(3).Infof("Expected Service resource, got: %v", attr.GetKind())
			return errors.NewInternalError(fmt.Errorf("Expected Service resource, got: %v", attr.GetKind()))
		}
		oldSvc = tmp
	}

	if isSubset(newSvc, oldSvc) {
		return nil
	}

	klog.V(4).Infof("Denying new use of ExternalIPs on Service %s/%s", newSvc.Namespace, newSvc.Name)
	return admission.NewForbidden(attr, fmt.Errorf("Use of external IPs is denied by admission control"))
}

func isSubset(newSvc, oldSvc *core.Service) bool {
	// If new has none, it's a subset.
	if len(newSvc.Spec.ExternalIPs) == 0 {
		return true
	}
	// If we have some but it's not an update, it's not a subset.
	if oldSvc == nil {
		return false
	}
	oldIPs := map[string]bool{}
	for _, ip := range oldSvc.Spec.ExternalIPs {
		oldIPs[ip] = true
	}
	// Every IP in newSvc must be in oldSvc
	for _, ip := range newSvc.Spec.ExternalIPs {
		if oldIPs[ip] == false {
			return false
		}
	}
	return true
}
