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

package install

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	core "k8s.io/kubernetes/federation/apis/core"
	core_v1 "k8s.io/kubernetes/federation/apis/core/v1"
	"k8s.io/kubernetes/pkg/api"
)

const importPrefix = "k8s.io/kubernetes/pkg/api"

var accessor = meta.NewAccessor()

// availableVersions lists all known external versions for this group from most preferred to least preferred
var availableVersions = []schema.GroupVersion{core_v1.SchemeGroupVersion}

func init() {
	api.Registry.RegisterVersions(availableVersions)
	externalVersions := []schema.GroupVersion{}
	for _, v := range availableVersions {
		if api.Registry.IsAllowedVersion(v) {
			externalVersions = append(externalVersions, v)
		}
	}
	if len(externalVersions) == 0 {
		glog.V(4).Infof("No version is registered for group %v", core.GroupName)
		return
	}

	if err := api.Registry.EnableVersions(externalVersions...); err != nil {
		glog.V(4).Infof("%v", err)
		return
	}
	if err := enableVersions(externalVersions); err != nil {
		glog.V(4).Infof("%v", err)
		return
	}
}

// TODO: enableVersions should be centralized rather than spread in each API
// group.
// We can combine api.Registry.RegisterVersions, api.Registry.EnableVersions and
// api.Registry.RegisterGroup once we have moved enableVersions there.
func enableVersions(externalVersions []schema.GroupVersion) error {
	addVersionsToScheme(externalVersions...)
	preferredExternalVersion := externalVersions[0]

	groupMeta := apimachinery.GroupMeta{
		GroupVersion:  preferredExternalVersion,
		GroupVersions: externalVersions,
		RESTMapper:    newRESTMapper(externalVersions),
		SelfLinker:    runtime.SelfLinker(accessor),
		InterfacesFor: interfacesFor,
	}

	if err := api.Registry.RegisterGroup(groupMeta); err != nil {
		return err
	}
	return nil
}

// userResources is a group of resources mostly used by a kubectl user
var userResources = []string{"svc"}

func newRESTMapper(externalVersions []schema.GroupVersion) meta.RESTMapper {
	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString(
		"Namespace",
	)

	// these kinds should be excluded from the list of resources
	ignoredKinds := sets.NewString(
		"ListOptions",
		"DeleteOptions",
		"Status")

	mapper := meta.NewDefaultRESTMapperFromScheme(externalVersions, interfacesFor, importPrefix, ignoredKinds, rootScoped, core.Scheme)
	// setup aliases for groups of resources
	mapper.AddResourceAlias("all", userResources...)

	return mapper
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func interfacesFor(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	switch version {
	case core_v1.SchemeGroupVersion:
		return &meta.VersionInterfaces{
			ObjectConvertor:  core.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		g, _ := api.Registry.Group(core.GroupName)
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, g.GroupVersions)
	}
}

func addVersionsToScheme(externalVersions ...schema.GroupVersion) {
	// add the internal version to Scheme
	if err := core.AddToScheme(core.Scheme); err != nil {
		// Programmer error, detect immediately
		panic(err)
	}
	// add the enabled external versions to Scheme
	for _, v := range externalVersions {
		if !api.Registry.IsEnabledVersion(v) {
			glog.Errorf("Version %s is not enabled, so it will not be added to the Scheme.", v)
			continue
		}
		switch v {
		case core_v1.SchemeGroupVersion:
			if err := core_v1.AddToScheme(core.Scheme); err != nil {
				// Programmer error, detect immediately
				panic(err)
			}
		}
	}
}
