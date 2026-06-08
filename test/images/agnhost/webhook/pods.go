/*
Copyright 2018 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"strings"

	"k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

const (
	podsInitContainerPatch string = `[
		 {"op":"add","path":"/spec/initContainers","value":[{"image":"webhook-added-image","name":"webhook-added-init-container","resources":{}}]}
	]`
	podsSidecarPatch string = `[
		{"op":"add", "path":"/spec/containers/-","value":{"image":"%v","name":"webhook-added-sidecar","resources":{}}}
	]`
)

// only allow pods to pull images from specific registry.
func admitPods(ar v1.AdmissionReview) *v1.AdmissionResponse {
	klog.V(2).Info("admitting pods")
	podResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	if ar.Request.Resource != podResource {
		err := fmt.Errorf("expect resource to be %s", podResource)
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}

	raw := ar.Request.Object.Raw
	pod := corev1.Pod{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(raw, nil, &pod); err != nil {
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}
	reviewResponse := v1.AdmissionResponse{}
	reviewResponse.Allowed = true

	var msg string
	if v, ok := pod.Labels["webhook-e2e-test"]; ok {
		if v == "webhook-disallow" {
			reviewResponse.Allowed = false
			msg = msg + "the pod contains unwanted label; "
		}
		if v == "wait-forever" {
			reviewResponse.Allowed = false
			msg = msg + "the pod response should not be sent; "
			<-make(chan int) // Sleep forever - no one sends to this channel
		}
	}
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, "webhook-disallow") {
			reviewResponse.Allowed = false
			msg = msg + "the pod contains unwanted container name; "
		}
	}
	if !reviewResponse.Allowed {
		reviewResponse.Result = &metav1.Status{Message: strings.TrimSpace(msg)}
	}
	return &reviewResponse
}

func mutatePods(ar v1.AdmissionReview) *v1.AdmissionResponse {
	shouldPatchPod := func(pod *corev1.Pod) bool {
		if pod.Name != "webhook-to-be-mutated" {
			return false
		}
		return !hasContainer(pod.Spec.InitContainers, "webhook-added-init-container")
	}
	return applyPodPatch(ar, shouldPatchPod, podsInitContainerPatch)
}

func mutatePodsSidecar(ar v1.AdmissionReview) *v1.AdmissionResponse {
	if sidecarImage == "" {
		return &v1.AdmissionResponse{
			Allowed: false,
			Result: &metav1.Status{
				Status:  "Failure",
				Message: "No image specified by the sidecar-image parameter",
				Code:    500,
			},
		}
	}
	shouldPatchPod := func(pod *corev1.Pod) bool {
		return !hasContainer(pod.Spec.Containers, "webhook-added-sidecar")
	}
	return applyPodPatch(ar, shouldPatchPod, fmt.Sprintf(podsSidecarPatch, sidecarImage))
}

func hasContainer(containers []corev1.Container, containerName string) bool {
	for _, container := range containers {
		if container.Name == containerName {
			return true
		}
	}
	return false
}

func applyPodPatch(ar v1.AdmissionReview, shouldPatchPod func(*corev1.Pod) bool, patch string) *v1.AdmissionResponse {
	klog.V(2).Info("mutating pods")
	podResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	if ar.Request.Resource != podResource {
		klog.Errorf("expect resource to be %s", podResource)
		return nil
	}

	raw := ar.Request.Object.Raw
	pod := corev1.Pod{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(raw, nil, &pod); err != nil {
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}
	reviewResponse := v1.AdmissionResponse{}
	reviewResponse.Allowed = true
	if shouldPatchPod(&pod) {
		reviewResponse.Patch = []byte(patch)
		pt := v1.PatchTypeJSONPatch
		reviewResponse.PatchType = &pt
	}
	return &reviewResponse
}

// denySpecificAttachment denies `kubectl attach to-be-attached-pod -i -c=container1"
// or equivalent client requests.
func denySpecificAttachment(ar v1.AdmissionReview) *v1.AdmissionResponse {
	klog.V(2).Info("handling attaching pods")
	if ar.Request.Name != "to-be-attached-pod" {
		return &v1.AdmissionResponse{Allowed: true}
	}
	podResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	if e, a := podResource, ar.Request.Resource; e != a {
		err := fmt.Errorf("expect resource to be %s, got %s", e, a)
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}
	if e, a := "attach", ar.Request.SubResource; e != a {
		err := fmt.Errorf("expect subresource to be %s, got %s", e, a)
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}

	raw := ar.Request.Object.Raw
	podAttachOptions := corev1.PodAttachOptions{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(raw, nil, &podAttachOptions); err != nil {
		klog.Error(err)
		return toV1AdmissionResponse(err)
	}
	klog.V(2).Info(fmt.Sprintf("podAttachOptions=%#v\n", podAttachOptions))
	if !podAttachOptions.Stdin || podAttachOptions.Container != "container1" {
		return &v1.AdmissionResponse{Allowed: true}
	}
	return &v1.AdmissionResponse{
		Allowed: false,
		Result: &metav1.Status{
			Message: "attaching to pod 'to-be-attached-pod' is not allowed",
		},
	}
}
