/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api/validation"
	extensionsvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

var ValidateClusterName = validation.NameIsDNSSubdomain

func ValidateClusterSpec(spec *federation.ClusterSpec, fieldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// address is required.
	if len(spec.ServerAddressByClientCIDRs) == 0 {
		allErrs = append(allErrs, field.Required(fieldPath.Child("serverAddressByClientCIDRs"), ""))
	}
	return allErrs
}

func ValidateCluster(cluster *federation.Cluster) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&cluster.ObjectMeta, false, ValidateClusterName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateClusterSpec(&cluster.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateClusterUpdate(cluster, oldCluster *federation.Cluster) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&cluster.ObjectMeta, &oldCluster.ObjectMeta, field.NewPath("metadata"))
	if cluster.Name != oldCluster.Name {
		allErrs = append(allErrs, field.Invalid(field.NewPath("meta", "name"),
			cluster.Name+" != "+oldCluster.Name, "cannot change cluster name"))
	}
	return allErrs
}

func ValidateClusterStatusUpdate(cluster, oldCluster *federation.Cluster) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&cluster.ObjectMeta, &oldCluster.ObjectMeta, field.NewPath("metadata"))
	return allErrs
}

func ValidateSubReplicaSet(rs *federation.SubReplicaSet) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&rs.ObjectMeta, true, extensionsvalidation.ValidateReplicaSetName, field.NewPath("metadata"))
	allErrs = append(allErrs, extensionsvalidation.ValidateReplicaSetSpec(&rs.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateSubReplicaSetUpdate(rs, oldRs *federation.SubReplicaSet) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&rs.ObjectMeta, &oldRs.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, extensionsvalidation.ValidateReplicaSetSpec(&rs.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateSubReplicaSetStatusUpdate(rs, oldRs *federation.SubReplicaSet) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&rs.ObjectMeta, &oldRs.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validation.ValidateNonnegativeField(int64(rs.Status.Replicas), field.NewPath("status", "replicas"))...)
	allErrs = append(allErrs, validation.ValidateNonnegativeField(int64(rs.Status.FullyLabeledReplicas), field.NewPath("status", "fullyLabeledReplicas"))...)
	allErrs = append(allErrs, validation.ValidateNonnegativeField(int64(rs.Status.ObservedGeneration), field.NewPath("status", "observedGeneration"))...)
	return allErrs
}
