package podsecurity

import (
	"context"
	"fmt"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	saadmission "k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
)

type SCCMutatingPodSpecExtractor struct {
	sccAdmission admission.MutationInterface
	delegate     podsecurityadmission.PodSpecExtractor
}

var SCCMutatingPodSpecExtractorInstance = &SCCMutatingPodSpecExtractor{
	delegate: podsecurityadmission.DefaultPodSpecExtractor{},
}

func (s *SCCMutatingPodSpecExtractor) SetSCCAdmission(sccAdmission admission.MutationInterface) {
	s.sccAdmission = sccAdmission
}

func (s *SCCMutatingPodSpecExtractor) HasPodSpec(gr schema.GroupResource) bool {
	return s.delegate.HasPodSpec(gr)
}

func (s *SCCMutatingPodSpecExtractor) ExtractPodSpec(obj runtime.Object) (*metav1.ObjectMeta, *corev1.PodSpec, error) {
	if s.sccAdmission == nil {
		return s.delegate.ExtractPodSpec(obj)
	}

	switch obj := obj.(type) {
	case *corev1.Pod:
		return s.delegate.ExtractPodSpec(obj)
	}

	podTemplateMeta, originalPodSpec, err := s.delegate.ExtractPodSpec(obj)
	if err != nil {
		return podTemplateMeta, originalPodSpec, err
	}
	if originalPodSpec == nil {
		return nil, nil, nil
	}
	objectMeta, err := meta.Accessor(obj)
	if err != nil {
		return podTemplateMeta, originalPodSpec, fmt.Errorf("unable to get metadata for SCC mutation: %w", err)
	}

	pod := &corev1.Pod{
		ObjectMeta: *podTemplateMeta.DeepCopy(),
		Spec:       *originalPodSpec.DeepCopy(),
	}
	if len(pod.Namespace) == 0 {
		pod.Namespace = objectMeta.GetNamespace()
	}
	if len(pod.Name) == 0 {
		pod.Name = "pod-for-container-named-" + objectMeta.GetName()
	}
	if len(pod.Spec.ServiceAccountName) == 0 {
		pod.Spec.ServiceAccountName = saadmission.DefaultServiceAccountName
	}
	internalPod := &core.Pod{}
	if err := v1.Convert_v1_Pod_To_core_Pod(pod, internalPod, nil); err != nil {
		return nil, nil, err
	}

	admissionAttributes := admission.NewAttributesRecord(
		internalPod,
		nil,
		corev1.SchemeGroupVersion.WithKind("Pod"),
		pod.Namespace,
		pod.Name,
		corev1.SchemeGroupVersion.WithResource("pods"),
		"",
		admission.Create,
		nil,
		false,
		&user.DefaultInfo{
			Name:   serviceaccount.MakeUsername(pod.Namespace, pod.Spec.ServiceAccountName),
			UID:    "",
			Groups: append([]string{user.AllAuthenticated}, serviceaccount.MakeGroupNames(pod.Namespace)...),
			Extra:  nil,
		})
	if err := s.sccAdmission.Admit(context.Background(), admissionAttributes, nil); err != nil {
		// don't fail the request, just warn if SCC will fail
		klog.ErrorS(err, "failed to mutate object for PSA using SCC")
		utilruntime.HandleError(fmt.Errorf("failed to mutate object for PSA using SCC: %w", err))
		// TODO remove this failure we're causing when SCC fails, but for now we actually need to see our test fail because that was almost really bad.
		return podTemplateMeta, originalPodSpec, nil
	}

	if err := v1.Convert_core_Pod_To_v1_Pod(internalPod, pod, nil); err != nil {
		return nil, nil, err
	}

	return podTemplateMeta, &pod.Spec, nil
}
