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

package options

import (
	"sort"
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

const (
	DefaultEtcdPathPrefix = "/registry"
)

// StorageSerializationOptions contains the options for encoding resources.
type StorageSerializationOptions struct {
	StorageVersions string
	// The default values for StorageVersions. StorageVersions overrides
	// these; you can change this if you want to change the defaults (e.g.,
	// for testing). This is not actually exposed as a flag.
	DefaultStorageVersions string
}

func NewStorageSerializationOptions() *StorageSerializationOptions {
	return &StorageSerializationOptions{
		DefaultStorageVersions: ToPreferredVersionString(legacyscheme.Scheme.PreferredVersionAllGroups()),
		StorageVersions:        ToPreferredVersionString(legacyscheme.Scheme.PreferredVersionAllGroups()),
	}
}

// StorageGroupsToEncodingVersion returns a map from group name to group version,
// computed from s.StorageVersions flag.
func (s *StorageSerializationOptions) StorageGroupsToEncodingVersion() (map[string]schema.GroupVersion, error) {
	storageVersionMap := map[string]schema.GroupVersion{}

	// First, get the defaults.
	if err := mergeGroupVersionIntoMap(s.DefaultStorageVersions, storageVersionMap); err != nil {
		return nil, err
	}
	// Override any defaults with the user settings.
	if err := mergeGroupVersionIntoMap(s.StorageVersions, storageVersionMap); err != nil {
		return nil, err
	}

	return storageVersionMap, nil
}

// dest must be a map of group to groupVersion.
func mergeGroupVersionIntoMap(gvList string, dest map[string]schema.GroupVersion) error {
	for _, gvString := range strings.Split(gvList, ",") {
		if gvString == "" {
			continue
		}
		// We accept two formats. "group/version" OR
		// "group=group/version". The latter is used when types
		// move between groups.
		if !strings.Contains(gvString, "=") {
			gv, err := schema.ParseGroupVersion(gvString)
			if err != nil {
				return err
			}
			dest[gv.Group] = gv

		} else {
			parts := strings.SplitN(gvString, "=", 2)
			gv, err := schema.ParseGroupVersion(parts[1])
			if err != nil {
				return err
			}
			dest[parts[0]] = gv
		}
	}

	return nil
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *StorageSerializationOptions) AddFlags(fs *pflag.FlagSet) {
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.
	fs.StringVar(&s.StorageVersions, "storage-versions", s.StorageVersions, ""+
		"The per-group version to store resources in. "+
		"Specified in the format \"group1/version1,group2/version2,...\". "+
		"In the case where objects are moved from one group to the other, "+
		"you may specify the format \"group1=group2/v1beta1,group3/v1beta1,...\". "+
		"You only need to pass the groups you wish to change from the defaults. "+
		"It defaults to a list of preferred versions of all known groups.")

	fs.MarkDeprecated("storage-versions", ""+
		"Please omit this flag to ensure the default storage versions are used ."+
		"Otherwise the cluster is not safe to upgrade to a version newer than 1.12. "+
		"This flag will be removed in 1.13.")

}

// ToPreferredVersionString returns the preferred versions of all registered
// groups in the form of "group1/version1,group2/version2,...".  This is compatible
// with the flag format
func ToPreferredVersionString(versions []schema.GroupVersion) string {
	var defaults []string
	for _, version := range versions {
		defaults = append(defaults, version.String())
	}
	// sorting provides stable output for help.
	sort.Strings(defaults)
	return strings.Join(defaults, ",")
}
