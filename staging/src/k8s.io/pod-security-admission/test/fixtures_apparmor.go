package test

import (
	"fmt"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/pod-security-admission/api"
)

const (
	permissionFormat = "container.apparmor.security.beta.kubernetes.io/%s"
)

func ensureAnnotation(pod *corev1.Pod) *corev1.Pod {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	return pod
}

func init() {
	appArmorFixture_1_0 := fixtureGenerator{
		expectErrorSubstring: "forbidden AppArmor profile",
		generateFail: func(pod *corev1.Pod) []*corev1.Pod {
			pod = ensureAnnotation(pod)
			return []*corev1.Pod{
				// container with no annotation
				tweak(pod, func(copy *corev1.Pod) {
					pod.Name = "test"
				}),

				// container with runtime/default annotation
				tweak(pod, func(copy *corev1.Pod) {
					pod.Name = "test"
					copy.Annotations[fmt.Sprintf(permissionFormat, pod.Name)] = "runtime/default"
				}),

				// container with localhost/foo annotation
				tweak(pod, func(copy *corev1.Pod) {
					pod.Name = "test"
					copy.Annotations[fmt.Sprintf(permissionFormat, pod.Name)] = "localhost/foo"
				}),

				// initContainer with no annotation
				tweak(pod, func(copy *corev1.Pod) {
					name := "init-container-test"
					pod.Spec.InitContainers = []corev1.Container{{Name: name}}
				}),

				// initContainer with runtime/default annotation
				tweak(pod, func(copy *corev1.Pod) {
					name := "init-container-test"
					copy.Annotations[fmt.Sprintf(permissionFormat, name)] = "runtime/default"
					pod.Spec.InitContainers = []corev1.Container{{Name: name}}
				}),

				// initContainer with localhost/foo annotation
				tweak(pod, func(copy *corev1.Pod) {
					name := "init-container-test"
					copy.Annotations[fmt.Sprintf(permissionFormat, name)] = "localhost/foo"
					pod.Spec.InitContainers = []corev1.Container{{Name: name}}
				}),
			}
		},
		generatePass: func(pod *corev1.Pod) []*corev1.Pod {
			pod = ensureAnnotation(pod)
			return []*corev1.Pod{
				// container with unconfined annotation
				tweak(pod, func(copy *corev1.Pod) {
					pod.Name = "test"
					copy.Annotations[fmt.Sprintf(permissionFormat, pod.Name)] = "unconfined"
				}),

				// initContainer with unconfined annotation
				tweak(pod, func(copy *corev1.Pod) {
					name := "init-container-test"
					pod.Spec.InitContainers = []corev1.Container{{Name: name}}
					copy.Annotations[fmt.Sprintf(permissionFormat, name)] = "unconfined"
				}),
			}
		},
	}

	registerFixtureGenerator(
		fixtureKey{level: api.LevelBaseline, version: api.MajorMinorVersion(1, 0), check: "appArmorProfile"},
		appArmorFixture_1_0,
	)
}
