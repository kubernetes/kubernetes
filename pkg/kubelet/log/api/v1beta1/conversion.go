package v1beta1

import (
	"unsafe"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/kubelet/log/api"
)

func Convert_v1beta1_PodLogPolicy_To_api_PodPolicy(in *PodLogPolicy, out *api.PodLogPolicy, s conversion.Scope) error {
	*out = *(*api.PodLogPolicy)(unsafe.Pointer(in))
	return nil
}
