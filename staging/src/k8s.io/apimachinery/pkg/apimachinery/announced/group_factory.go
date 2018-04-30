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

// GroupMetaFactory has the logic for actually assembling and registering a group.
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
// the registered group & versions. Then it calls register.
func (gmf *GroupMetaFactory) Register(m *registered.APIRegistrationManager, scheme *runtime.Scheme) error {
	if gmf.GroupArgs == nil {
		return fmt.Errorf("partially registered groups are not allowed, only got versions: %#v", gmf.VersionArgs)
	}
	if len(gmf.VersionArgs) == 0 {
		return fmt.Errorf("group %v registered but no versions registered", gmf.GroupArgs.GroupName)
	}
	if m.IsRegistered(gmf.GroupArgs.GroupName) {
		return fmt.Errorf("the group %q has already been registered.", gmf.GroupArgs.GroupName)
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
		return fmt.Errorf("group %v has versions in the priority list that were never registered: %s", gmf.GroupArgs.GroupName, pvSet)
	}
	prioritizedVersions = append(prioritizedVersions, unprioritizedVersions...)
	m.RegisterVersions(prioritizedVersions)
	gmf.prioritizedVersionList = prioritizedVersions

	externalVersions := []schema.GroupVersion{}
	for _, v := range gmf.prioritizedVersionList {
		externalVersions = append(externalVersions, v)
		gmf.VersionArgs[v.Version].AddToScheme(scheme)
	}
	if len(externalVersions) == 0 {
		glog.V(4).Infof("No version is registered for group %v", gmf.GroupArgs.GroupName)
		return nil
	}

	if gmf.GroupArgs.AddInternalObjectsToScheme != nil {
		gmf.GroupArgs.AddInternalObjectsToScheme(scheme)
	}

	groupMeta := &apimachinery.GroupMeta{
		GroupVersions: externalVersions,
	}

	if err := m.RegisterGroup(*groupMeta); err != nil {
		return err
	}
	return nil
}
