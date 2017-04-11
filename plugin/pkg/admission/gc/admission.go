/*
Copyright 2016 The Kubernetes Authors.

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

package gc

import (
	"fmt"
	"io"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func init() {
	admission.RegisterPlugin("OwnerReferencesPermissionEnforcement", func(config io.Reader) (admission.Interface, error) {
		return &gcPermissionsEnforcement{
			Handler: admission.NewHandler(admission.Create, admission.Update),
		}, nil
	})
}

// gcPermissionsEnforcement is an implementation of admission.Interface.
type gcPermissionsEnforcement struct {
	*admission.Handler

	authorizer authorizer.Authorizer
}

func (a *gcPermissionsEnforcement) Admit(attributes admission.Attributes) (err error) {
	// if we aren't changing owner references, then the edit is always allowed
	if !isChangingOwnerReference(attributes.GetObject(), attributes.GetOldObject()) {
		return nil
	}

	deleteAttributes := authorizer.AttributesRecord{
		User:            attributes.GetUserInfo(),
		Verb:            "delete",
		Namespace:       attributes.GetNamespace(),
		APIGroup:        attributes.GetResource().Group,
		APIVersion:      attributes.GetResource().Version,
		Resource:        attributes.GetResource().Resource,
		Subresource:     attributes.GetSubresource(),
		Name:            attributes.GetName(),
		ResourceRequest: true,
		Path:            "",
	}
	allowed, reason, err := a.authorizer.Authorize(deleteAttributes)
	if allowed {
		return nil
	}

	return admission.NewForbidden(attributes, fmt.Errorf("cannot set an ownerRef on a resource you can't delete: %v, %v", reason, err))
}

func isChangingOwnerReference(newObj, oldObj runtime.Object) bool {
	newMeta, err := meta.Accessor(newObj)
	if err != nil {
		// if we don't have objectmeta, we don't have the object reference
		return false
	}

	if oldObj == nil {
		return len(newMeta.GetOwnerReferences()) > 0
	}
	oldMeta, err := meta.Accessor(oldObj)
	if err != nil {
		// if we don't have objectmeta, we don't have the object reference
		return false
	}

	// compare the old and new.  If they aren't the same, then we're trying to change an ownerRef
	oldOwners := oldMeta.GetOwnerReferences()
	newOwners := newMeta.GetOwnerReferences()
	if len(oldOwners) != len(newOwners) {
		return true
	}
	for i := range oldOwners {
		if !apiequality.Semantic.DeepEqual(oldOwners[i], newOwners[i]) {
			return true
		}
	}

	return false
}

func (a *gcPermissionsEnforcement) SetAuthorizer(authorizer authorizer.Authorizer) {
	a.authorizer = authorizer
}

func (a *gcPermissionsEnforcement) Validate() error {
	if a.authorizer == nil {
		return fmt.Errorf("missing authorizer")
	}
	return nil
}
