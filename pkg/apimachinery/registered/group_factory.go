/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package registered

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

type GroupVersionFactory struct {
	GroupName   string
	VersionName string

	// MetadataAccessor meta.MetadataAccessor   // nil for default
	// ObjectConvertor  runtime.ObjectConvertor // nil for default

	AddToScheme func(*runtime.Scheme)
}

type GroupMetaFactory struct {
	GroupName              string
	VersionPreferenceOrder []string
	ImportPrefix           string

	RootScopedKinds sets.String // nil is allowed
	IgnoredKinds    sets.String // nil is allowed

	// May be nil if none
	AddInternalObjectsToScheme func(*runtime.Scheme)
}

type groupAndVersions struct {
	group *GroupMetaFactory
	// map of version name to version factory
	versions map[string]*GroupVersionFactory

	// assembled by Register()
	prioritizedVersionList []unversioned.GroupVersion
}

var (
	groupFactory = make(groupFactories)

	// Call AnnounceGroupVersion in the setup code for your version.
	AnnounceGroupVersion      = groupFactory.AnnounceGroupVersion
	AnnounceGroup             = groupFactory.AnnounceGroup
	RegisterAnnouncedVersions = groupFactory.RegisterAndEnableAll
)

type groupFactories map[string]*groupAndVersions

func (gfm groupFactories) group(groupName string) *groupAndVersions {
	gav, ok := gfm[groupName]
	if !ok {
		gav = &groupAndVersions{versions: map[string]*GroupVersionFactory{}}
		gfm[groupName] = gav
	}
	return gav
}

func (gfm groupFactories) AnnounceGroupVersion(gvf *GroupVersionFactory) error {
	gav := gfm.group(gvf.GroupName)
	if _, ok := gav.versions[gvf.VersionName]; ok {
		return fmt.Errorf("version %q in group %q has already been announced", gvf.VersionName, gvf.GroupName)
	}
	gav.versions[gvf.VersionName] = gvf
	return nil
}

func (self groupFactories) AnnounceGroup(gmf *GroupMetaFactory) error {
	gav := self.group(gmf.GroupName)
	if gav.group != nil {
		return fmt.Errorf("group %q has already been announced", gmf.GroupName)
	}
	gav.group = gmf
	return nil
}

func (gfm groupFactories) RegisterAndEnableAll(m *Manager, scheme *runtime.Scheme) error {
	for groupName, gav := range gfm {
		if err := gav.Register(m); err != nil {
			return fmt.Errorf("error registering %v: %v", groupName, err)
		}
		if err := gav.Enable(m, scheme); err != nil {
			return fmt.Errorf("error enabling %v: %v", groupName, err)
		}
	}
	return nil
}

// Register constructs the finalized prioritized version list and sanity checks
// the announced group & versions. Then it calls register.
func (gav *groupAndVersions) Register(m *Manager) error {
	if gav.group == nil {
		return fmt.Errorf("partially announced groups are not allowed, only got versions: %#v", gav.versions)
	}
	if len(gav.versions) == 0 {
		return fmt.Errorf("group %v announced but no versions announced", gav.group.GroupName)
	}

	pvSet := sets.NewString(gav.group.VersionPreferenceOrder...)
	if pvSet.Len() != len(gav.group.VersionPreferenceOrder) {
		return fmt.Errorf("preference order for group %v has duplicates: %v", gav.group.GroupName, gav.group.VersionPreferenceOrder)
	}
	prioritizedVersions := []unversioned.GroupVersion{}
	for _, v := range gav.group.VersionPreferenceOrder {
		prioritizedVersions = append(
			prioritizedVersions,
			unversioned.GroupVersion{
				gav.group.GroupName,
				v,
			},
		)
	}

	availableVersions := []unversioned.GroupVersion{}
	for _, v := range gav.versions {
		if v.GroupName != gav.group.GroupName {
			return fmt.Errorf("found %v/%v in group %v?", v.GroupName, v.VersionName, gav.group.GroupName)
		}
		if pvSet.Has(v.VersionName) {
			pvSet.Delete(v.VersionName)
			continue
		}
		availableVersions = append(availableVersions, unversioned.GroupVersion{v.GroupName, v.VersionName})
	}
	if len(availableVersions) != 0 {
		glog.Warningf("group %v has unprioritized versions %#v. They will have an arbitrary preference order!", gav.group.GroupName, availableVersions)
	}
	if pvSet.Len() != 0 {
		return fmt.Errorf("group %v has versions in the priority list that were never announced: %s", gav.group.GroupName, pvSet)
	}
	prioritizedVersions = append(prioritizedVersions, availableVersions...)
	m.RegisterVersions(prioritizedVersions)
	gav.prioritizedVersionList = prioritizedVersions
	return nil
}

func (gav *groupAndVersions) newRESTMapper(externalVersions []unversioned.GroupVersion, groupMeta *apimachinery.GroupMeta) meta.RESTMapper {
	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString()
	if gav.group.RootScopedKinds != nil {
		rootScoped = gav.group.RootScopedKinds
	}
	ignoredKinds := sets.NewString()
	if gav.group.IgnoredKinds != nil {
		ignoredKinds = gav.group.IgnoredKinds
	}

	return api.NewDefaultRESTMapper(
		externalVersions,
		groupMeta.InterfacesFor,
		gav.group.ImportPrefix,
		ignoredKinds,
		rootScoped,
	)
}

// Enable enables group versions that are allowed, adds methods to the scheme, etc.
func (gav *groupAndVersions) Enable(m *Manager, scheme *runtime.Scheme) error {
	externalVersions := []unversioned.GroupVersion{}
	for _, v := range gav.prioritizedVersionList {
		if !m.IsAllowedVersion(v) {
			continue
		}
		externalVersions = append(externalVersions, v)
		if err := m.EnableVersions(v); err != nil {
			return err
		}
		gav.versions[v.Version].AddToScheme(scheme)
	}
	if len(externalVersions) == 0 {
		glog.V(4).Infof("No version is registered for group %v", gav.group.GroupName)
		return nil
	}

	if gav.group.AddInternalObjectsToScheme != nil {
		gav.group.AddInternalObjectsToScheme(scheme)
	}

	preferredExternalVersion := externalVersions[0]
	accessor := meta.NewAccessor()

	groupMeta := &apimachinery.GroupMeta{
		GroupVersion:  preferredExternalVersion,
		GroupVersions: externalVersions,
		SelfLinker:    runtime.SelfLinker(accessor),
	}
	groupMeta.RESTMapper = gav.newRESTMapper(externalVersions, groupMeta)
	for _, gvf := range gav.versions {
		err := groupMeta.AddVersion(
			unversioned.GroupVersion{gvf.GroupName, gvf.VersionName},
			&meta.VersionInterfaces{
				ObjectConvertor:  scheme,
				MetadataAccessor: accessor,
			},
		)
		if err != nil {
			return err
		}
	}

	if err := m.RegisterGroup(*groupMeta); err != nil {
		return err
	}
	api.RegisterRESTMapper(groupMeta.RESTMapper)
	return nil
}
