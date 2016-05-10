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

// Package install installs the componentconfig API group, making it available as
// an option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/componentconfig"

	// TODO: Remove from here, add to places where this install package is imported.
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1/install"
)

func init() {
	if err := registered.AnnounceGroup(&registered.GroupMetaFactory{
		GroupName:                  "componentconfig",
		VersionPreferenceOrder:     []string{"v1alpha1"},
		ImportPrefix:               "k8s.io/kubernetes/pkg/apis/componentconfig",
		AddInternalObjectsToScheme: componentconfig.AddToScheme,
	}); err != nil {
		panic(err)
	}
}
