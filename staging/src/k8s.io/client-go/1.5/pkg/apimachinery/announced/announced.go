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

// Package announced contains tools for announcing API group factories. This is
// distinct from registration (in the 'registered' package) in that it's safe
// to announce every possible group linked in, but only groups requested at
// runtime should be registered. This package contains both a registry, and
// factory code (which was formerly copy-pasta in every install package).
package announced

import (
	"fmt"

	"k8s.io/client-go/1.5/pkg/apimachinery/registered"
	"k8s.io/client-go/1.5/pkg/runtime"
)

var (
	DefaultGroupFactoryRegistry = make(APIGroupFactoryRegistry)

	// These functions will announce your group or version.
	AnnounceGroupVersion = DefaultGroupFactoryRegistry.AnnounceGroupVersion
	AnnounceGroup        = DefaultGroupFactoryRegistry.AnnounceGroup
)

// APIGroupFactoryRegistry allows for groups and versions to announce themselves,
// which simply makes them available and doesn't take other actions. Later,
// users of the registry can select which groups and versions they'd actually
// like to register with an APIRegistrationManager.
//
// (Right now APIRegistrationManager has separate 'registration' and 'enabled'
// concepts-- APIGroupFactory is going to take over the former function;
// they will overlap untill the refactoring is finished.)
//
// The key is the group name. After initialization, this should be treated as
// read-only. It is implemented as a map from group name to group factory, and
// it is safe to use this knowledge to manually pick out groups to register
// (e.g., for testing).
type APIGroupFactoryRegistry map[string]*GroupMetaFactory

func (gar APIGroupFactoryRegistry) group(groupName string) *GroupMetaFactory {
	gmf, ok := gar[groupName]
	if !ok {
		gmf = &GroupMetaFactory{VersionArgs: map[string]*GroupVersionFactoryArgs{}}
		gar[groupName] = gmf
	}
	return gmf
}

// AnnounceGroupVersion adds the particular arguments for this group version to the group factory.
func (gar APIGroupFactoryRegistry) AnnounceGroupVersion(gvf *GroupVersionFactoryArgs) error {
	gmf := gar.group(gvf.GroupName)
	if _, ok := gmf.VersionArgs[gvf.VersionName]; ok {
		return fmt.Errorf("version %q in group %q has already been announced", gvf.VersionName, gvf.GroupName)
	}
	gmf.VersionArgs[gvf.VersionName] = gvf
	return nil
}

// AnnounceGroup adds the group-wide arguments to the group factory.
func (gar APIGroupFactoryRegistry) AnnounceGroup(args *GroupMetaFactoryArgs) error {
	gmf := gar.group(args.GroupName)
	if gmf.GroupArgs != nil {
		return fmt.Errorf("group %q has already been announced", args.GroupName)
	}
	gmf.GroupArgs = args
	return nil
}

// RegisterAndEnableAll throws every factory at the specified API registration
// manager, and lets it decide which to register. (If you want to do this a la
// cart, you may look through gar itself-- it's just a map.)
func (gar APIGroupFactoryRegistry) RegisterAndEnableAll(m *registered.APIRegistrationManager, scheme *runtime.Scheme) error {
	for groupName, gmf := range gar {
		if err := gmf.Register(m); err != nil {
			return fmt.Errorf("error registering %v: %v", groupName, err)
		}
		if err := gmf.Enable(m, scheme); err != nil {
			return fmt.Errorf("error enabling %v: %v", groupName, err)
		}
	}
	return nil
}

// AnnouncePreconstructedFactory announces a factory which you've manually assembled.
// You may call this instead of calling AnnounceGroup and AnnounceGroupVersion.
func (gar APIGroupFactoryRegistry) AnnouncePreconstructedFactory(gmf *GroupMetaFactory) error {
	name := gmf.GroupArgs.GroupName
	if _, exists := gar[name]; exists {
		return fmt.Errorf("the group %q has already been announced.", name)
	}
	gar[name] = gmf
	return nil
}
