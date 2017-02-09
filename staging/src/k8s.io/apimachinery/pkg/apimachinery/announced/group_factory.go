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

package announced

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

type SchemeFunc func(*runtime.Scheme) error
type VersionToSchemeFunc map[string]SchemeFunc

// GroupVersionFactoryArgs contains all the per-version parts of a GroupMetaFactory.
type GroupVersionFactoryArgs struct {
	GroupName   string
	VersionName string

	AddToScheme SchemeFunc
}

// GroupMetaFactoryArgs contains the group-level args of a GroupMetaFactory.
type GroupMetaFactoryArgs struct {
	// GroupName is the name of the API-Group
	//
	// example: 'servicecatalog.k8s.io'
	GroupName              string
	VersionPreferenceOrder []string
	// ImportPrefix is the base go package of the API-Group
	//
	// example: 'k8s.io/kubernetes/pkg/apis/autoscaling'
	ImportPrefix string
	// RootScopedKinds are resources that are not namespaced.
	RootScopedKinds sets.String // nil is allowed
	IgnoredKinds    sets.String // nil is allowed

	// May be nil if there are no internal objects.
	AddInternalObjectsToScheme SchemeFunc
}

// NewGroupMetaFactory builds the args for you. This is for if you're
// constructing a factory all at once and not using the registry.
func NewGroupMetaFactory(groupArgs *GroupMetaFactoryArgs, versions VersionToSchemeFunc) *GroupMetaFactory {
	gmf := &GroupMetaFactory{
		GroupArgs:   groupArgs,
		VersionArgs: map[string]*GroupVersionFactoryArgs{},
	}
	for v, f := range versions {
		gmf.VersionArgs[v] = &GroupVersionFactoryArgs{
			GroupName:   groupArgs.GroupName,
			VersionName: v,
			AddToScheme: f,
		}
	}
	return gmf
}

// Announce adds this Group factory to the global factory registry. It should
// only be called if you constructed the GroupMetaFactory yourself via
// NewGroupMetadFactory.
// Note that this will panic on an error, since it's expected that you'll be
// calling this at initialization time and any error is a result of a
// programmer importing the wrong set of packages. If this assumption doesn't
// work for you, just call DefaultGroupFactoryRegistry.AnnouncePreconstructedFactory
// yourself.
func (gmf *GroupMetaFactory) Announce(groupFactoryRegistry APIGroupFactoryRegistry) *GroupMetaFactory {
	if err := groupFactoryRegistry.AnnouncePreconstructedFactory(gmf); err != nil {
		panic(err)
	}
	return gmf
}

// GroupMetaFactory has the logic for actually assembling and registering a group.
//
// There are two ways of obtaining one of these.
// 1. You can announce your group and versions separately, and then let the
//    GroupFactoryRegistry assemble this object for you. (This allows group and
//    versions to be imported separately, without referencing each other, to
//    keep import trees small.)
// 2. You can call NewGroupMetaFactory(), which is mostly a drop-in replacement
//    for the old, bad way of doing things. You can then call .Announce() to
//    announce your constructed factory to any code that would like to do
//    things the new, better way.
//
// Note that GroupMetaFactory actually does construct GroupMeta objects, but
// currently it does so in a way that's very entangled with an
// APIRegistrationManager. It's a TODO item to cleanly separate that interface.
type GroupMetaFactory struct {
	GroupArgs *GroupMetaFactoryArgs
	// map of version name to version factory
	VersionArgs map[string]*GroupVersionFactoryArgs

	// assembled by Register()
	prioritizedVersionList []schema.GroupVersion
}

// Register constructs the finalized prioritized version list and sanity checks
// the announced group & versions. Then it calls register.
func (gmf *GroupMetaFactory) Register(m *registered.APIRegistrationManager) error {
	if gmf.GroupArgs == nil {
		return fmt.Errorf("partially announced groups are not allowed, only got versions: %#v", gmf.VersionArgs)
	}
	if len(gmf.VersionArgs) == 0 {
		return fmt.Errorf("group %v announced but no versions announced", gmf.GroupArgs.GroupName)
	}

	pvSet := sets.NewString(gmf.GroupArgs.VersionPreferenceOrder...)
	if pvSet.Len() != len(gmf.GroupArgs.VersionPreferenceOrder) {
		return fmt.Errorf("preference order for group %v has duplicates: %v", gmf.GroupArgs.GroupName, gmf.GroupArgs.VersionPreferenceOrder)
	}
	prioritizedVersions := []schema.GroupVersion{}
	for _, v := range gmf.GroupArgs.VersionPreferenceOrder {
		prioritizedVersions = append(
			prioritizedVersions,
			schema.GroupVersion{
				Group:   gmf.GroupArgs.GroupName,
				Version: v,
			},
		)
	}

	// Go through versions that weren't explicitly prioritized.
	unprioritizedVersions := []schema.GroupVersion{}
	for _, v := range gmf.VersionArgs {
		if v.GroupName != gmf.GroupArgs.GroupName {
			return fmt.Errorf("found %v/%v in group %v?", v.GroupName, v.VersionName, gmf.GroupArgs.GroupName)
		}
		if pvSet.Has(v.VersionName) {
			pvSet.Delete(v.VersionName)
			continue
		}
		unprioritizedVersions = append(unprioritizedVersions, schema.GroupVersion{Group: v.GroupName, Version: v.VersionName})
	}
	if len(unprioritizedVersions) > 1 {
		glog.Warningf("group %v has multiple unprioritized versions: %#v. They will have an arbitrary preference order!", gmf.GroupArgs.GroupName, unprioritizedVersions)
	}
	if pvSet.Len() != 0 {
		return fmt.Errorf("group %v has versions in the priority list that were never announced: %s", gmf.GroupArgs.GroupName, pvSet)
	}
	prioritizedVersions = append(prioritizedVersions, unprioritizedVersions...)
	m.RegisterVersions(prioritizedVersions)
	gmf.prioritizedVersionList = prioritizedVersions
	return nil
}

func (gmf *GroupMetaFactory) newRESTMapper(scheme *runtime.Scheme, externalVersions []schema.GroupVersion, groupMeta *apimachinery.GroupMeta) meta.RESTMapper {
	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString()
	if gmf.GroupArgs.RootScopedKinds != nil {
		rootScoped = gmf.GroupArgs.RootScopedKinds
	}
	ignoredKinds := sets.NewString()
	if gmf.GroupArgs.IgnoredKinds != nil {
		ignoredKinds = gmf.GroupArgs.IgnoredKinds
	}

	return meta.NewDefaultRESTMapperFromScheme(
		externalVersions,
		groupMeta.InterfacesFor,
		gmf.GroupArgs.ImportPrefix,
		ignoredKinds,
		rootScoped,
		scheme,
	)
}

// Enable enables group versions that are allowed, adds methods to the scheme, etc.
func (gmf *GroupMetaFactory) Enable(m *registered.APIRegistrationManager, scheme *runtime.Scheme) error {
	externalVersions := []schema.GroupVersion{}
	for _, v := range gmf.prioritizedVersionList {
		if !m.IsAllowedVersion(v) {
			continue
		}
		externalVersions = append(externalVersions, v)
		if err := m.EnableVersions(v); err != nil {
			return err
		}
		gmf.VersionArgs[v.Version].AddToScheme(scheme)
	}
	if len(externalVersions) == 0 {
		glog.V(4).Infof("No version is registered for group %v", gmf.GroupArgs.GroupName)
		return nil
	}

	if gmf.GroupArgs.AddInternalObjectsToScheme != nil {
		gmf.GroupArgs.AddInternalObjectsToScheme(scheme)
	}

	preferredExternalVersion := externalVersions[0]
	accessor := meta.NewAccessor()

	groupMeta := &apimachinery.GroupMeta{
		GroupVersion:  preferredExternalVersion,
		GroupVersions: externalVersions,
		SelfLinker:    runtime.SelfLinker(accessor),
	}
	for _, v := range externalVersions {
		gvf := gmf.VersionArgs[v.Version]
		if err := groupMeta.AddVersionInterfaces(
			schema.GroupVersion{Group: gvf.GroupName, Version: gvf.VersionName},
			&meta.VersionInterfaces{
				ObjectConvertor:  scheme,
				MetadataAccessor: accessor,
			},
		); err != nil {
			return err
		}
	}
	groupMeta.InterfacesFor = groupMeta.DefaultInterfacesFor
	groupMeta.RESTMapper = gmf.newRESTMapper(scheme, externalVersions, groupMeta)

	if err := m.RegisterGroup(*groupMeta); err != nil {
		return err
	}
	return nil
}

// RegisterAndEnable is provided only to allow this code to get added in multiple steps.
// It's really bad that this is called in init() methods, but supporting this
// temporarily lets us do the change incrementally.
func (gmf *GroupMetaFactory) RegisterAndEnable(registry *registered.APIRegistrationManager, scheme *runtime.Scheme) error {
	if err := gmf.Register(registry); err != nil {
		return err
	}
	if err := gmf.Enable(registry, scheme); err != nil {
		return err
	}

	return nil
}
