/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/api/admission/v1alpha1"
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

// InstallWithInternal registers the API group with attached internal types and adds both kinds of types to a scheme.
// This is only used internally by Kubernetes and not needed for the use of the versioned types alone.
func InstallWithInternal(groupFactoryRegistry announced.APIGroupFactoryRegistry, registry *registered.APIRegistrationManager, scheme *runtime.Scheme, addInternal announced.SchemeFunc) {
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:                  v1alpha1.GroupName,
			VersionPreferenceOrder:     []string{v1alpha1.SchemeGroupVersion.Version},
			RootScopedKinds:            sets.NewString("AdmissionReview"),
			AddInternalObjectsToScheme: addInternal,
		},
		announced.VersionToSchemeFunc{
			v1alpha1.SchemeGroupVersion.Version: v1alpha1.AddToScheme,
		},
	).Announce(groupFactoryRegistry).RegisterAndEnable(registry, scheme); err != nil {
		panic(err)
	}
}

// Install registers the API group and adds types to a scheme
func Install(groupFactoryRegistry announced.APIGroupFactoryRegistry, registry *registered.APIRegistrationManager, scheme *runtime.Scheme) {
	InstallWithInternal(groupFactoryRegistry, registry, scheme, nil)
}
