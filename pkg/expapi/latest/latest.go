/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package latest

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/registered"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/expapi/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var (
	Version  string
	Versions []string

	accessor   = meta.NewAccessor()
	Codec      runtime.Codec
	SelfLinker = runtime.SelfLinker(accessor)
	RESTMapper meta.RESTMapper
)

const importPrefix = "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi"

func init() {
	Version = registered.RegisteredVersions[0]
	Codec = runtime.CodecFor(api.Scheme, Version)
	// Put the registered versions in Versions in reverse order.
	for i := len(registered.RegisteredVersions) - 1; i >= 0; i-- {
		Versions = append(Versions, registered.RegisteredVersions[i])
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := util.NewStringSet()

	ignoredKinds := util.NewStringSet()

	RESTMapper = api.NewDefaultRESTMapper(Versions, InterfacesFor, importPrefix, ignoredKinds, rootScoped)
	api.RegisterRESTMapper(RESTMapper)
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(version string) (*meta.VersionInterfaces, error) {
	switch version {
	case "v1":
		return &meta.VersionInterfaces{
			Codec:            v1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %s)", version, strings.Join(Versions, ", "))
	}
}
