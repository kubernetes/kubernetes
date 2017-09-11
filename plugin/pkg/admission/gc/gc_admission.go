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
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("OwnerReferencesPermissionEnforcement", func(config io.Reader) (admission.Interface, error) {
		// the pods/status endpoint is ignored by this plugin since old kubelets
		// corrupt them.  the pod status strategy ensures status updates cannot mutate
		// ownerRef.
		whiteList := []whiteListItem{
			{
				groupResource: schema.GroupResource{Resource: "pods"},
				subresource:   "status",
			},
		}
		return &gcPermissionsEnforcement{
			Handler:   admission.NewHandler(admission.Create, admission.Update),
			whiteList: whiteList,
		}, nil
	})
}

// gcPermissionsEnforcement is an implementation of admission.Interface.
type gcPermissionsEnforcement struct {
	*admission.Handler

	authorizer authorizer.Authorizer

	restMapper meta.RESTMapper

	// items in this whitelist are ignored upon admission.
	// any item in this list must protect against ownerRef mutations
	// via strategy enforcement.
	whiteList []whiteListItem
}

// whiteListItem describes an entry in a whitelist ignored by gc permission enforcement.
type whiteListItem struct {
	groupResource schema.GroupResource
	subresource   string
}

// isWhiteListed returns true if the specified item is in the whitelist.
func (a *gcPermissionsEnforcement) isWhiteListed(groupResource schema.GroupResource, subresource string) bool {
	for _, item := range a.whiteList {
		if item.groupResource == groupResource && item.subresource == subresource {
			return true
		}
	}
	return false
}

func (a *gcPermissionsEnforcement) Admit(attributes admission.Attributes) (err error) {
	// // if the request is in the whitelist, we skip mutation checks for this resource.
	if a.isWhiteListed(attributes.GetResource().GroupResource(), attributes.GetSubresource()) {
		return nil
	}

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

	// Further check if the user is setting ownerReference.blockOwnerDeletion to
	// true. If so, only allows the change if the user has delete permission of
	// the _OWNER_
	newBlockingRefs := newBlockingOwnerDeletionRefs(attributes.GetObject(), attributes.GetOldObject())
	for _, ref := range newBlockingRefs {
		records, err := a.ownerRefToDeleteAttributeRecords(ref, attributes)
		if err != nil {
			return admission.NewForbidden(attributes, fmt.Errorf("cannot set blockOwnerDeletion in this case because cannot find RESTMapping for APIVersion %s Kind %s: %v, %v", ref.APIVersion, ref.Kind, reason, err))
		}
		// Multiple records are returned if ref.Kind could map to multiple
		// resources. User needs to have delete permission on all the
		// matched Resources.
		for _, record := range records {
			allowed, reason, err := a.authorizer.Authorize(record)
			if !allowed {
				return admission.NewForbidden(attributes, fmt.Errorf("cannot set blockOwnerDeletion if an ownerReference refers to a resource you can't set finalizers on: %v, %v", reason, err))
			}
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

// Translates ref to a DeleteAttribute deleting the object referred by the ref.
// OwnerReference only records the object kind, which might map to multiple
// resources, so multiple DeleteAttribute might be returned.
func (a *gcPermissionsEnforcement) ownerRefToDeleteAttributeRecords(ref metav1.OwnerReference, attributes admission.Attributes) ([]authorizer.AttributesRecord, error) {
	var ret []authorizer.AttributesRecord
	groupVersion, err := schema.ParseGroupVersion(ref.APIVersion)
	if err != nil {
		return ret, err
	}
	mappings, err := a.restMapper.RESTMappings(schema.GroupKind{Group: groupVersion.Group, Kind: ref.Kind}, groupVersion.Version)
	if err != nil {
		return ret, err
	}
	for _, mapping := range mappings {
		ret = append(ret, authorizer.AttributesRecord{
			User: attributes.GetUserInfo(),
			Verb: "update",
			// ownerReference can only refer to an object in the same namespace, so attributes.GetNamespace() equals to the owner's namespace
			Namespace:       attributes.GetNamespace(),
			APIGroup:        groupVersion.Group,
			APIVersion:      groupVersion.Version,
			Resource:        mapping.Resource,
			Subresource:     "finalizers",
			Name:            ref.Name,
			ResourceRequest: true,
			Path:            "",
		})
	}
	return ret, nil
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

// Returns new blocking ownerReferences, and references whose blockOwnerDeletion
// field is changed from nil or false to true.
func newBlockingOwnerDeletionRefs(newObj, oldObj runtime.Object) []metav1.OwnerReference {
	newMeta, err := meta.Accessor(newObj)
	if err != nil {
		// if we don't have objectmeta, we don't have the object reference
		return nil
	}
	newRefs := newMeta.GetOwnerReferences()
	blockingNewRefs := blockingOwnerRefs(newRefs)
	if len(blockingNewRefs) == 0 {
		return nil
	}

	if oldObj == nil {
		return blockingNewRefs
	}
	oldMeta, err := meta.Accessor(oldObj)
	if err != nil {
		// if we don't have objectmeta, treat it as if all the ownerReference are newly created
		return blockingNewRefs
	}

	var ret []metav1.OwnerReference
	indexedOldRefs := indexByUID(oldMeta.GetOwnerReferences())
	for _, ref := range blockingNewRefs {
		oldRef, ok := indexedOldRefs[ref.UID]
		if !ok {
			// if ref is newly added, and it's blocking, then returns it.
			ret = append(ret, ref)
			continue
		}
		wasNotBlocking := oldRef.BlockOwnerDeletion == nil || *oldRef.BlockOwnerDeletion == false
		if wasNotBlocking {
			ret = append(ret, ref)
		}
	}
	return ret
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
	if a.restMapper == nil {
		return fmt.Errorf("missing restMapper")
	}
	return nil
}
