package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	storagemigrationv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/storagemigration"
)

func Convert_v1alpha1_MigrationCondition_To_v1_Condition(in *storagemigrationv1alpha1.MigrationCondition, out *metav1.Condition, s conversion.Scope) error {
	out.Type = string(in.Type)
	out.Status = metav1.ConditionStatus(in.Status)
	out.LastTransitionTime = in.LastUpdateTime
	out.Reason = in.Reason
	out.Message = in.Message
	return nil
}

func Convert_v1_Condition_To_v1alpha1_MigrationCondition(in *metav1.Condition, out *storagemigrationv1alpha1.MigrationCondition, s conversion.Scope) error {
	out.Type = storagemigrationv1alpha1.MigrationConditionType(in.Type)
	out.Status = corev1.ConditionStatus(in.Status)
	out.LastUpdateTime = in.LastTransitionTime
	out.Reason = in.Reason
	out.Message = in.Message
	return nil
}

func Convert_v1alpha1_StorageVersionMigrationSpec_To_storagemigration_StorageVersionMigrationSpec(in *storagemigrationv1alpha1.StorageVersionMigrationSpec, out *storagemigration.StorageVersionMigrationSpec, s conversion.Scope) error {
	return Convert_v1alpha1_GroupVersionResource_To_storagemigration_GroupResource(&in.Resource, &out.Resource, s)
}

func Convert_v1alpha1_GroupVersionResource_To_storagemigration_GroupResource(in *storagemigrationv1alpha1.GroupVersionResource, out *storagemigration.GroupResource, s conversion.Scope) error {
	out.Group = in.Group
	out.Resource = in.Resource
	return nil
}

func Convert_storagemigration_GroupResource_To_v1alpha1_GroupVersionResource(in *storagemigration.GroupResource, out *storagemigrationv1alpha1.GroupVersionResource, s conversion.Scope) error {
	out.Group = in.Group
	out.Resource = in.Resource
	return nil
}
