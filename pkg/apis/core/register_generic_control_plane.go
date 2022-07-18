/*
Copyright 2020 The Kubernetes Authors.

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

package core

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	// GenericControlPlaneGroupName is the name of the group when installed in the generic control plane
	GenericControlPlaneGroupName = "core"

	// GenericControlPlaneSchemeGroupVersion is group version used to register these objects
	GenericControlPlaneSchemeGroupVersion = schema.GroupVersion{Group: GenericControlPlaneGroupName, Version: "v1"}

	// GenericControlPlaneSchemeBuilder object to register various known types for the control plane
	GenericControlPlaneSchemeBuilder = runtime.NewSchemeBuilder(addGenericControlPlaneKnownTypes)

	// AddToGenericControlPlaneScheme represents a func that can be used to apply all the registered
	// funcs in a scheme
	AddToGenericControlPlaneScheme = GenericControlPlaneSchemeBuilder.AddToScheme
)

func addGenericControlPlaneKnownTypes(scheme *runtime.Scheme) error {
	if err := scheme.AddIgnoredConversionType(&metav1.TypeMeta{}, &metav1.TypeMeta{}); err != nil {
		return err
	}
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Event{},
		&EventList{},
		&List{},
		&LimitRange{},
		&LimitRangeList{},
		&ResourceQuota{},
		&ResourceQuotaList{},
		&Namespace{},
		&NamespaceList{},
		&ServiceAccount{},
		&ServiceAccountList{},
		&Secret{},
		&SecretList{},
		&SerializedReference{},
		&RangeAllocation{},
		&ConfigMap{},
		&ConfigMapList{},
	)

	scheme.AddKnownTypes(GenericControlPlaneSchemeGroupVersion,
		&Event{},
		&EventList{},
		&List{},
		&LimitRange{},
		&LimitRangeList{},
		&ResourceQuota{},
		&ResourceQuotaList{},
		&Namespace{},
		&NamespaceList{},
		&ServiceAccount{},
		&ServiceAccountList{},
		&Secret{},
		&SecretList{},
		&SerializedReference{},
		&RangeAllocation{},
		&ConfigMap{},
		&ConfigMapList{},
	)

	return nil
}
