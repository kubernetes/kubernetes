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
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/latest"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

func init() {
	// this means that the entire group is disabled
	if len(latest.ExternalVersions) == 0 {
		return
	}

	groupMeta := latest.GroupMeta{
		GroupVersion:  latest.PreferredExternalVersion,
		GroupVersions: latest.ExternalVersions,
		Codec:         latest.Codec,
		RESTMapper:    latest.RESTMapper,
		SelfLinker:    runtime.SelfLinker(latest.Accessor),
		InterfacesFor: latest.InterfacesFor,
	}

	if err := latest.RegisterGroup(groupMeta); err != nil {
		glog.V(4).Infof("%v", err)
		return
	}

	api.RegisterRESTMapper(groupMeta.RESTMapper)
}
