package selinux

import (
	corev1 "k8s.io/api/core/v1"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	corev1conversions "k8s.io/kubernetes/pkg/apis/core/v1"
)

func ToInternalSELinuxOptions(external *corev1.SELinuxOptions) (*coreapi.SELinuxOptions, error) {
	if external == nil {
		return nil, nil
	}
	internal := &coreapi.SELinuxOptions{}
	err := corev1conversions.Convert_v1_SELinuxOptions_To_core_SELinuxOptions(external, internal, nil)
	return internal, err
}

func ToInternalSELinuxOptionsOrDie(external *corev1.SELinuxOptions) *coreapi.SELinuxOptions {
	ret, err := ToInternalSELinuxOptions(external)
	if err != nil {
		panic(err)
	}
	return ret
}
