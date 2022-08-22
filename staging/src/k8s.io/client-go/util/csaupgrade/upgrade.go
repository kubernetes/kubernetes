package csaupgrade

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// Upgrades the Manager information for fields managed with CSA
// Prepares fields owned by `csaManager` for 'Update' operations for use now
// with the given `ssaManager` for `Apply` operations
//
// csaManager - Name of FieldManager formerly used for `Update` operations
// ssaManager - Name of FieldManager formerly used for `Apply` operations
// subResource - Name of subresource used for api calls or empty string for main resource
func UpgradeManagedFields(
	obj runtime.Object,
	csaManager string,
	ssaManager string,
	subResource string,
) (runtime.Object, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, fmt.Errorf("error accessing object metadata: %w", err)
	}

	managed, error := fieldmanager.DecodeManagedFields(accessor.GetManagedFields())
	if error != nil {
		return nil, fmt.Errorf("failed to decode managed fields: %w", error)
	}
	// If SSA manager  exists:
	//		find CSA manager of same version, union. discard the rest
	// Else SSA manager does not exist:
	//		find most recent CSA manager. convert to Apply operation

	ssaIdentifier, err := fieldmanager.BuildManagerIdentifier(&metav1.ManagedFieldsEntry{
		Manager:     ssaManager,
		Operation:   metav1.ManagedFieldsOperationApply,
		Subresource: subResource,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to build manager identifier for ssa manager")
	}

	ssaMan, ssaExists := managed.Fields()[ssaIdentifier]

	// Collect all relevant CSA managers before operating on them
	csaManagers := map[string]fieldpath.VersionedSet{}
	for name, entry := range managed.Fields() {
		if entry.Applied() {
			// Not interested in SSA managed fields entries
			continue
		}

		// Manager string is a JSON representation of encoded entry
		// Pull manager name and subresource from it
		encodedVersionedSet := &metav1.ManagedFieldsEntry{}
		err = json.Unmarshal([]byte(name), encodedVersionedSet)
		if err != nil {
			return nil, fmt.Errorf("error unmarshalling manager identifier %v: %v", name, err)
		}

		if encodedVersionedSet.Manager != csaManager ||
			encodedVersionedSet.Subresource != subResource {
			continue
		}

		csaManagers[name] = entry
	}

	if len(csaManagers) == 0 {
		return obj, nil
	}

	if ssaExists {
		for name, entry := range csaManagers {
			if entry.APIVersion() == ssaMan.APIVersion() {
				// Merge entries if they are compatible versions
				ssaMan = fieldpath.NewVersionedSet(
					ssaMan.Set().Union(entry.Set()),
					entry.APIVersion(),
					true,
				)
				managed.Fields()[ssaIdentifier] = ssaMan
			}

			// Discard entry in all cases:
			//	if it has the wrong version we discard since managed fields versions
			//		cannot be converted
			//	if it has the correct version its fields were moved into the
			//		ssaManager's fieldSet
			delete(managed.Fields(), name)
		}
	} else {
		// Loop through sorted CSA managers. Take the first one we care about
		firstName := ""
		for _, entry := range accessor.GetManagedFields() {
			if entry.Manager == csaManager &&
				entry.Subresource == subResource &&
				entry.Operation == metav1.ManagedFieldsOperationUpdate {

				if len(firstName) == 0 {
					ident, err := fieldmanager.BuildManagerIdentifier(&entry)
					if err != nil {
						return nil, fmt.Errorf("failed to build manager identifier: %w", err)
					}

					firstName = ident
					break
				}
			}
		}

		managed.Fields()[ssaIdentifier] = csaManagers[firstName]

		for name := range csaManagers {
			delete(managed.Fields(), name)
		}
	}

	now := metav1.Now()
	managed.Times()[ssaIdentifier] = &now

	copied := obj.DeepCopyObject()
	if err := fieldmanager.EncodeObjectManagedFields(copied, managed); err != nil {
		return nil, err
	}
	return copied, nil
}
