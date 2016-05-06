/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package install installs the v1 monolithic api, making it available as an
// option to all of the API encoding/decoding machinery.
package install

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

const importPrefix = "k8s.io/kubernetes/pkg/api"

var accessor = meta.NewAccessor()

// availableVersions lists all known external versions for this group from most preferred to least preferred
var availableVersions = []unversioned.GroupVersion{v1.SchemeGroupVersion}

func init() {
	registered.RegisterVersions(availableVersions)
	externalVersions := []unversioned.GroupVersion{}
	for _, v := range availableVersions {
		if registered.IsAllowedVersion(v) {
			externalVersions = append(externalVersions, v)
		}
	}
	if len(externalVersions) == 0 {
		glog.V(4).Infof("No version is registered for group %v", api.GroupName)
		return
	}

	if err := registered.EnableVersions(externalVersions...); err != nil {
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
// We can combine registered.RegisterVersions, registered.EnableVersions and
// registered.RegisterGroup once we have moved enableVersions there.
func enableVersions(externalVersions []unversioned.GroupVersion) error {
	addVersionsToScheme(externalVersions...)
	preferredExternalVersion := externalVersions[0]

	groupMeta := apimachinery.GroupMeta{
		GroupVersion:  preferredExternalVersion,
		GroupVersions: externalVersions,
		RESTMapper:    newRESTMapper(externalVersions),
		SelfLinker:    runtime.SelfLinker(accessor),
		InterfacesFor: interfacesFor,
	}

	if err := registered.RegisterGroup(groupMeta); err != nil {
		return err
	}
	api.RegisterRESTMapper(groupMeta.RESTMapper)
	return nil
}

// userResources is a group of resources mostly used by a kubectl user
var userResources = []string{"rc", "svc", "pods", "pvc"}

func newRESTMapper(externalVersions []unversioned.GroupVersion) meta.RESTMapper {
	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString(
		"Node",
		"Namespace",
		"PersistentVolume",
		"ComponentStatus",
	)

	// these kinds should be excluded from the list of resources
	ignoredKinds := sets.NewString(
		"ListOptions",
		"DeleteOptions",
		"Status",
		"PodLogOptions",
		"PodExecOptions",
		"PodAttachOptions",
		"PodProxyOptions",
		"NodeProxyOptions",
		"ServiceProxyOptions",
		"ThirdPartyResource",
		"ThirdPartyResourceData",
		"ThirdPartyResourceList")

	mapper := api.NewDefaultRESTMapper(externalVersions, interfacesFor, importPrefix, ignoredKinds, rootScoped)
	// setup aliases for groups of resources
	mapper.AddResourceAlias("all", userResources...)

	return mapper
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func interfacesFor(version unversioned.GroupVersion) (*meta.VersionInterfaces, error) {
	switch version {
	case v1.SchemeGroupVersion:
		return &meta.VersionInterfaces{
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		g, _ := registered.Group(api.GroupName)
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, g.GroupVersions)
	}
}

func addVersionsToScheme(externalVersions ...unversioned.GroupVersion) {
	// add the internal version to Scheme
	api.AddToScheme(api.Scheme)
	// add the enabled external versions to Scheme
	for _, v := range externalVersions {
		if !registered.IsEnabledVersion(v) {
			glog.Errorf("Version %s is not enabled, so it will not be added to the Scheme.", v)
			continue
		}
		switch v {
		case v1.SchemeGroupVersion:
			v1.AddToScheme(api.Scheme)
		}
	}

	// This is a "fast-path" that avoids reflection for common types. It focuses on the objects that are
	// converted the most in the cluster.
	// TODO: generate one of these for every external API group - this is to prove the impact
	api.Scheme.AddGenericConversionFunc(func(objA, objB interface{}, s conversion.Scope) (bool, error) {
		switch a := objA.(type) {
		case *v1.Pod:
			switch b := objB.(type) {
			case *api.Pod:
				return true, v1.Convert_v1_Pod_To_api_Pod(a, b, s)
			}
		case *api.Pod:
			switch b := objB.(type) {
			case *v1.Pod:
				return true, v1.Convert_api_Pod_To_v1_Pod(a, b, s)
			}

		case *v1.Event:
			switch b := objB.(type) {
			case *api.Event:
				return true, v1.Convert_v1_Event_To_api_Event(a, b, s)
			}
		case *api.Event:
			switch b := objB.(type) {
			case *v1.Event:
				return true, v1.Convert_api_Event_To_v1_Event(a, b, s)
			}

		case *v1.ReplicationController:
			switch b := objB.(type) {
			case *api.ReplicationController:
				return true, v1.Convert_v1_ReplicationController_To_api_ReplicationController(a, b, s)
			}
		case *api.ReplicationController:
			switch b := objB.(type) {
			case *v1.ReplicationController:
				return true, v1.Convert_api_ReplicationController_To_v1_ReplicationController(a, b, s)
			}

		case *v1.Node:
			switch b := objB.(type) {
			case *api.Node:
				return true, v1.Convert_v1_Node_To_api_Node(a, b, s)
			}
		case *api.Node:
			switch b := objB.(type) {
			case *v1.Node:
				return true, v1.Convert_api_Node_To_v1_Node(a, b, s)
			}

		case *v1.Namespace:
			switch b := objB.(type) {
			case *api.Namespace:
				return true, v1.Convert_v1_Namespace_To_api_Namespace(a, b, s)
			}
		case *api.Namespace:
			switch b := objB.(type) {
			case *v1.Namespace:
				return true, v1.Convert_api_Namespace_To_v1_Namespace(a, b, s)
			}

		case *v1.Service:
			switch b := objB.(type) {
			case *api.Service:
				return true, v1.Convert_v1_Service_To_api_Service(a, b, s)
			}
		case *api.Service:
			switch b := objB.(type) {
			case *v1.Service:
				return true, v1.Convert_api_Service_To_v1_Service(a, b, s)
			}

		case *v1.Endpoints:
			switch b := objB.(type) {
			case *api.Endpoints:
				return true, v1.Convert_v1_Endpoints_To_api_Endpoints(a, b, s)
			}
		case *api.Endpoints:
			switch b := objB.(type) {
			case *v1.Endpoints:
				return true, v1.Convert_api_Endpoints_To_v1_Endpoints(a, b, s)
			}
		}
		return false, nil
	})
}
