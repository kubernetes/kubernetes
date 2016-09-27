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

// Package install installs the batch API group, making it available as
// an option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/client-go/1.5/pkg/apimachinery/announced"
	"k8s.io/client-go/1.5/pkg/apis/batch"
	"k8s.io/client-go/1.5/pkg/apis/batch/v1"
	"k8s.io/client-go/1.5/pkg/apis/batch/v2alpha1"
)

func init() {
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:                  batch.GroupName,
			VersionPreferenceOrder:     []string{v1.SchemeGroupVersion.Version, v2alpha1.SchemeGroupVersion.Version},
			ImportPrefix:               "k8s.io/client-go/1.5/pkg/apis/batch",
			AddInternalObjectsToScheme: batch.AddToScheme,
		},
		announced.VersionToSchemeFunc{
			v1.SchemeGroupVersion.Version:       v1.AddToScheme,
			v2alpha1.SchemeGroupVersion.Version: v2alpha1.AddToScheme,
		},
	).Announce().RegisterAndEnable(); err != nil {
		panic(err)
	}
}
