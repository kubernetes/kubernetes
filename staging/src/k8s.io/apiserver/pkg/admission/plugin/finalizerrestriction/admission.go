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

package finalizerrestriction

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/klog"
)

const (
	// PluginName indicates the name of admission plugin
	PluginName = "FinalizerRestriction"
)

// Register registers a plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return &FinalizerRestriction{}, nil
	})
}

// FinalizerRestriction is an implementation of admission Interface.
// It restricts on who can add/remove finalizer on resources.
type FinalizerRestriction struct {
	*admission.Handler

	authorizer authorizer.Authorizer
}

var _ admission.Interface = &FinalizerRestriction{}
var _ initializer.WantsAuthorizer = &FinalizerRestriction{}

// SetAuthorizer injects the authorizer
func (r *FinalizerRestriction) SetAuthorizer(authorizer authorizer.Authorizer) {
	r.authorizer = authorizer
}

// ValidateInitialization verifies the authorizer injection.
func (r *FinalizerRestriction) ValidateInitialization() error {
	if r.authorizer == nil {
		return fmt.Errorf("missing authorizer")
	}
	return nil
}

// Admit decides whether the finalizer can be added or removed.
func (r *FinalizerRestriction) Admit(a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetOperation() != admission.Create && a.GetOperation() != admission.Update {
		return nil
	}
	if a.GetSubresource() != "" {
		return nil
	}

	obj, err := meta.Accessor(a.GetObject())
	if err != nil {
		klog.V(2).Infof("can not get meta accessor: %s", err)
		return nil
	}

	newFinalizers := sets.NewString(obj.GetFinalizers()...)
	oldFinalizers := sets.NewString()

	if a.GetOldObject() != nil {
		oldObj, err := meta.Accessor(a.GetOldObject())
		if err != nil {
			klog.V(2).Infof("can not get meta accessor for old object: %s", err)
			return nil
		}
		oldFinalizers.Insert(oldObj.GetFinalizers()...)
	}

	union := newFinalizers.Union(oldFinalizers)
	intersection := newFinalizers.Intersection(oldFinalizers)

	for _, finalizer := range union.Difference(intersection).List() {
		attr := authorizer.AttributesRecord{
			User:            a.GetUserInfo(),
			Verb:            "finalize",
			Namespace:       a.GetNamespace(),
			Name:            finalizer,
			APIGroup:        a.GetResource().Group,
			Resource:        a.GetResource().Resource,
			ResourceRequest: true,
		}
		decision, reason, err := r.authorizer.Authorize(attr)
		if err != nil {
			klog.V(5).Infof("cannot authorize for finalizing %s: %s, %s", finalizer, reason, err)
		}
		if decision == authorizer.DecisionAllow {
			continue
		}
		return admission.NewForbidden(a, fmt.Errorf("not allowed to finalize %s", finalizer))
	}

	return nil
}
