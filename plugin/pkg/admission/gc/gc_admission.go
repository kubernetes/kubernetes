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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

func init() {
	kubeapiserveradmission.Plugins.Register("OwnerReferencesPermissionEnforcement", func(config io.Reader) (admission.Interface, error) {
		return &gcPermissionsEnforcement{
			Handler: admission.NewHandler(admission.Create, admission.Update),
		}, nil
	})
}

// gcPermissionsEnforcement is an implementation of admission.Interface.
type gcPermissionsEnforcement struct {
	*admission.Handler

	authorizer authorizer.Authorizer

	restMapper meta.RESTMapper
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
	if !allowed {
		return admission.NewForbidden(attributes, fmt.Errorf("cannot set an ownerRef on a resource you can't delete: %v, %v", reason, err))
	}

	// further check if the user is setting ownerReference.blockOwnerDeletion,
	// and if so, only allows the change if the user has delete permission of
	// the _OWNER_
	changedRefs := changingBlockOwnerDeletion(attributes.GetObject(), attributes.GetOldObject())
	for _, ref := range changedRefs {
		attribute, err := a.toDeleteAttribute(ref, attributes)
		if err != nil {
			// An error occurs if ref.APIVersion and ref.Kind cannot be parsed.
			// If it's caused by a non-kubernetes core API, currently garbage
			// collector doesn't handle such object, so it's ok to admit changes
			// to such owner ref. Other errors should be prevented by the
			// validation code.
			continue
		}
		allowed, reason, err := a.authorizer.Authorize(attribute)
		if !allowed {
			return admission.NewForbidden(attributes, fmt.Errorf("cannot set blockOwnerDeletion in an ownerReference refers to a resource you can't delete: %v, %v", reason, err))
		}
	}

	return nil

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

// translate ref to a DeleteAttribute deleting the object ref refers to.
func (a *gcPermissionsEnforcement) toDeleteAttribute(ref metav1.OwnerReference, attributes admission.Attributes) (authorizer.AttributesRecord, error) {
	groupVersion, err := schema.ParseGroupVersion(ref.APIVersion)
	if err != nil {
		return authorizer.AttributesRecord{}, err
	}
	mapping, err := a.restMapper.RESTMapping(schema.GroupKind{groupVersion.Group, ref.Kind}, groupVersion.Version)
	if err != nil {
		return authorizer.AttributesRecord{}, err
	}
	return authorizer.AttributesRecord{
		User: attributes.GetUserInfo(),
		Verb: "delete",
		// ownerReference can only refer to an object in the same namespace, so attributes.GetnNamespace() equals to the owner's namespace
		Namespace:       attributes.GetNamespace(),
		APIGroup:        groupVersion.Group,
		APIVersion:      groupVersion.Version,
		Resource:        mapping.Resource,
		Name:            ref.Name,
		ResourceRequest: true,
		Path:            "",
	}, nil
}

// only keeps the blocking refs
func blockingOwnerRefs(refs []metav1.OwnerReference) []metav1.OwnerReference {
	var ret []metav1.OwnerReference
	for _, ref := range refs {
		if ref.BlockOwnerDeletion != nil && *ref.BlockOwnerDeletion == true {
			ret = append(ret, ref)
		}
	}
	return ret
}

func indexByUID(refs []metav1.OwnerReference) map[types.UID]metav1.OwnerReference {
	ret := make(map[types.UID]metav1.OwnerReference)
	for _, ref := range refs {
		ret[ref.UID] = ref
	}
	return ret
}

// Returns new blocking ownerReferences, or references whose blockOwnerDeletion field is changed.
// Changes between nil and false are ignored.
func changingBlockOwnerDeletion(newObj, oldObj runtime.Object) []metav1.OwnerReference {
	newMeta, err := meta.Accessor(newObj)
	if err != nil {
		// if we don't have objectmeta, we don't have the object reference
		return nil
	}
	newRefs := newMeta.GetOwnerReferences()

	if oldObj == nil {
		return blockingOwnerRefs(newRefs)
	}
	oldMeta, err := meta.Accessor(oldObj)
	if err != nil {
		// if we don't have objectmeta, treat it as if all the ownerReference are newly created
		return blockingOwnerRefs(newRefs)
	}

	var changedRefs []metav1.OwnerReference
	indexedOldRefs := indexByUID(oldMeta.GetOwnerReferences())
	for _, newRef := range newRefs {
		newBlocking := newRef.BlockOwnerDeletion != nil && *newRef.BlockOwnerDeletion == true
		if oldRef, ok := indexedOldRefs[newRef.UID]; ok {
			oldBlocking := oldRef.BlockOwnerDeletion != nil && *oldRef.BlockOwnerDeletion == true
			if oldBlocking != newBlocking {
				changedRefs = append(changedRefs, newRef)
			}
		} else if newBlocking {
			// if newRef is newly added, and it's blocking, also return it.
			changedRefs = append(changedRefs, newRef)
		}
	}
	return changedRefs
}

func (a *gcPermissionsEnforcement) SetAuthorizer(authorizer authorizer.Authorizer) {
	a.authorizer = authorizer
}

func (a *gcPermissionsEnforcement) SetRESTMapper(restMapper meta.RESTMapper) {
	a.restMapper = restMapper
}

func (a *gcPermissionsEnforcement) Validate() error {
	if a.authorizer == nil {
		return fmt.Errorf("missing authorizer")
	}
	return nil
}
