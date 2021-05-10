/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	api "k8s.io/kubernetes/pkg/apis/core"
	v1defaults "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/controller"
	kubelet "k8s.io/kubernetes/pkg/kubelet/types"
	utilpointer "k8s.io/utils/pointer"
)

func TestIgnoresNonCreate(t *testing.T) {
	for _, op := range []admission.Operation{admission.Delete, admission.Connect} {
		handler := NewServiceAccount()
		if handler.Handles(op) {
			t.Errorf("Expected not to handle operation %s", op)
		}
	}
}

func TestIgnoresNonPodResource(t *testing.T) {
	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("CustomResource").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	handler := admissiontesting.WithReinvocationTesting(t, NewServiceAccount())
	err := handler.Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Expected non-pod resource allowed, got err: %v", err)
	}
}

func TestIgnoresNilObject(t *testing.T) {
	attrs := admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	handler := admissiontesting.WithReinvocationTesting(t, NewServiceAccount())
	err := handler.Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Expected nil object allowed allowed, got err: %v", err)
	}
}

func TestIgnoresNonPodObject(t *testing.T) {
	obj := &api.Namespace{}
	attrs := admission.NewAttributesRecord(obj, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	handler := admissiontesting.WithReinvocationTesting(t, NewServiceAccount())
	err := handler.Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Expected non pod object allowed, got err: %v", err)
	}
}

func TestIgnoresMirrorPod(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				kubelet.ConfigMirrorAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, NewServiceAccount()).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Expected mirror pod without service account or secrets allowed, got err: %v", err)
	}
}

func TestRejectsMirrorPodWithServiceAccount(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				kubelet.ConfigMirrorAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			ServiceAccountName: "default",
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, NewServiceAccount()).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a service account")
	}
}

func TestRejectsMirrorPodWithSecretVolumes(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				kubelet.ConfigMirrorAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "mysecret"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, NewServiceAccount()).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a secret volume")
	}
}

func TestRejectsMirrorPodWithServiceAccountTokenVolumeProjections(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				kubelet.ConfigMirrorAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{
					Projected: &api.ProjectedVolumeSource{
						Sources: []api.VolumeProjection{{ServiceAccountToken: &api.ServiceAccountTokenProjection{}}},
					},
				},
				},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, NewServiceAccount()).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a ServiceAccountToken volume projection")
	}
}

func TestAssignsDefaultServiceAccountAndBoundTokenWithNoSecretTokens(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true

	// Add the default service account for the ns into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	v1PodIn := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{}},
		},
	}
	v1defaults.SetObjectDefaults_Pod(v1PodIn)
	pod := &api.Pod{}
	if err := v1defaults.Convert_v1_Pod_To_core_Pod(v1PodIn, pod, nil); err != nil {
		t.Fatal(err)
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Fatalf("Expected success, got: %v", err)
	}

	expectedVolumes := []api.Volume{{
		Name: "cleared",
		VolumeSource: api.VolumeSource{
			Projected: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					{ServiceAccountToken: &api.ServiceAccountTokenProjection{ExpirationSeconds: 3607, Path: "token"}},
					{ConfigMap: &api.ConfigMapProjection{LocalObjectReference: api.LocalObjectReference{Name: "kube-root-ca.crt"}, Items: []api.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}}}},
					{DownwardAPI: &api.DownwardAPIProjection{Items: []api.DownwardAPIVolumeFile{{Path: "namespace", FieldRef: &api.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.namespace"}}}}},
					{ConfigMap: &api.ConfigMapProjection{LocalObjectReference: api.LocalObjectReference{Name: "openshift-service-ca.crt"}, Items: []api.KeyToPath{{Key: "service-ca.crt", Path: "service-ca.crt"}}}},
				},
				DefaultMode: utilpointer.Int32(0644),
			},
		},
	}}
	expectedVolumeMounts := []api.VolumeMount{{
		Name:      "cleared",
		ReadOnly:  true,
		MountPath: "/var/run/secrets/kubernetes.io/serviceaccount",
	}}

	// clear generated volume names
	for i := range pod.Spec.Volumes {
		if len(pod.Spec.Volumes[i].Name) > 0 {
			pod.Spec.Volumes[i].Name = "cleared"
		}
	}
	for i := range pod.Spec.Containers[0].VolumeMounts {
		if len(pod.Spec.Containers[0].VolumeMounts[i].Name) > 0 {
			pod.Spec.Containers[0].VolumeMounts[i].Name = "cleared"
		}
	}

	if !reflect.DeepEqual(expectedVolumes, pod.Spec.Volumes) {
		t.Errorf("unexpected volumes: %s", cmp.Diff(expectedVolumes, pod.Spec.Volumes))
	}
	if !reflect.DeepEqual(expectedVolumeMounts, pod.Spec.Containers[0].VolumeMounts) {
		t.Errorf("unexpected volumes: %s", cmp.Diff(expectedVolumeMounts, pod.Spec.Containers[0].VolumeMounts))
	}

	// ensure result converted to v1 matches defaulted object
	v1PodOut := &corev1.Pod{}
	if err := v1defaults.Convert_core_Pod_To_v1_Pod(pod, v1PodOut, nil); err != nil {
		t.Fatal(err)
	}
	v1PodOutDefaulted := v1PodOut.DeepCopy()
	v1defaults.SetObjectDefaults_Pod(v1PodOutDefaulted)
	if !reflect.DeepEqual(v1PodOut, v1PodOutDefaulted) {
		t.Error(cmp.Diff(v1PodOut, v1PodOutDefaulted))
	}
}

func TestFetchesUncachedServiceAccount(t *testing.T) {
	ns := "myns"

	// Build a test client that the admission plugin can use to look up the service account missing from its cache
	client := fake.NewSimpleClientset(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.client = client

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
}

func TestDeniesInvalidServiceAccount(t *testing.T) {
	ns := "myns"

	// Build a test client that the admission plugin can use to look up the service account missing from its cache
	client := fake.NewSimpleClientset()

	admit := NewServiceAccount()
	admit.SetExternalKubeClientSet(client)
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected error for missing service account, got none")
	}
}

func TestAutomountsAPIToken(t *testing.T) {

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.generateName = testGenerateName
	admit.MountServiceAccountToken = true

	ns := "myns"
	serviceAccountName := DefaultServiceAccountName
	serviceAccountUID := "12345"

	tokenName := generatedVolumeName

	expectedVolume := api.Volume{
		Name: tokenName,
		VolumeSource: api.VolumeSource{
			Projected: TokenVolumeSource(),
		},
	}
	expectedVolumeMount := api.VolumeMount{
		Name:      tokenName,
		ReadOnly:  true,
		MountPath: DefaultAPITokenMountPath,
	}
	// Add the default service account for the ns with a token into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
	if len(pod.Spec.Volumes) != 1 {
		t.Fatalf("Expected 1 volume, got %d", len(pod.Spec.Volumes))
	}
	if !reflect.DeepEqual(expectedVolume, pod.Spec.Volumes[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolume, pod.Spec.Volumes[0])
	}
	if len(pod.Spec.Containers[0].VolumeMounts) != 1 {
		t.Fatalf("Expected 1 volume mount, got %d", len(pod.Spec.Containers[0].VolumeMounts))
	}
	if !reflect.DeepEqual(expectedVolumeMount, pod.Spec.Containers[0].VolumeMounts[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolumeMount, pod.Spec.Containers[0].VolumeMounts[0])
	}

	// testing InitContainers
	pod = &api.Pod{
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
	if len(pod.Spec.Volumes) != 1 {
		t.Fatalf("Expected 1 volume, got %d", len(pod.Spec.Volumes))
	}
	if !reflect.DeepEqual(expectedVolume, pod.Spec.Volumes[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolume, pod.Spec.Volumes[0])
	}
	if len(pod.Spec.InitContainers[0].VolumeMounts) != 1 {
		t.Fatalf("Expected 1 volume mount, got %d", len(pod.Spec.InitContainers[0].VolumeMounts))
	}
	if !reflect.DeepEqual(expectedVolumeMount, pod.Spec.InitContainers[0].VolumeMounts[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolumeMount, pod.Spec.InitContainers[0].VolumeMounts[0])
	}
}

func TestRespectsExistingMount(t *testing.T) {
	ns := "myns"
	serviceAccountName := DefaultServiceAccountName
	serviceAccountUID := "12345"

	expectedVolumeMount := api.VolumeMount{
		Name:      "my-custom-mount",
		ReadOnly:  false,
		MountPath: DefaultAPITokenMountPath,
	}

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true

	// Add the default service account for the ns with a token into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
	})

	// Define a pod with a container that already mounts a volume at the API token path
	// Admission should respect that
	// Additionally, no volume should be created if no container is going to use it
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					VolumeMounts: []api.VolumeMount{
						expectedVolumeMount,
					},
				},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
	if len(pod.Spec.Volumes) != 0 {
		t.Fatalf("Expected 0 volumes (shouldn't create a volume for a secret we don't need), got %d", len(pod.Spec.Volumes))
	}
	if len(pod.Spec.Containers[0].VolumeMounts) != 1 {
		t.Fatalf("Expected 1 volume mount, got %d", len(pod.Spec.Containers[0].VolumeMounts))
	}
	if !reflect.DeepEqual(expectedVolumeMount, pod.Spec.Containers[0].VolumeMounts[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolumeMount, pod.Spec.Containers[0].VolumeMounts[0])
	}

	// check init containers
	pod = &api.Pod{
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{
					VolumeMounts: []api.VolumeMount{
						expectedVolumeMount,
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
	if len(pod.Spec.Volumes) != 0 {
		t.Fatalf("Expected 0 volumes (shouldn't create a volume for a secret we don't need), got %d", len(pod.Spec.Volumes))
	}
	if len(pod.Spec.InitContainers[0].VolumeMounts) != 1 {
		t.Fatalf("Expected 1 volume mount, got %d", len(pod.Spec.InitContainers[0].VolumeMounts))
	}
	if !reflect.DeepEqual(expectedVolumeMount, pod.Spec.InitContainers[0].VolumeMounts[0]) {
		t.Fatalf("Expected\n\t%#v\ngot\n\t%#v", expectedVolumeMount, pod.Spec.InitContainers[0].VolumeMounts[0])
	}
}

func TestAllowsReferencedSecret(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		Secrets: []corev1.ObjectReference{
			{Name: "foo"},
		},
	})

	pod1 := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod1, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "container-1",
					Env: []api.EnvVar{
						{
							Name: "env-1",
							ValueFrom: &api.EnvVarSource{
								SecretKeyRef: &api.SecretKeySelector{
									LocalObjectReference: api.LocalObjectReference{Name: "foo"},
								},
							},
						},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "container-1",
					EnvFrom: []api.EnvFromSource{
						{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{
					Name: "container-1",
					Env: []api.EnvVar{
						{
							Name: "env-1",
							ValueFrom: &api.EnvVarSource{
								SecretKeyRef: &api.SecretKeySelector{
									LocalObjectReference: api.LocalObjectReference{Name: "foo"},
								},
							},
						},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{
					Name: "container-1",
					EnvFrom: []api.EnvFromSource{
						{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			EphemeralContainers: []api.EphemeralContainer{
				{
					EphemeralContainerCommon: api.EphemeralContainerCommon{
						Name: "container-2",
						Env: []api.EnvVar{
							{
								Name: "env-1",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{Name: "foo"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	// validate enforces restrictions on secret mounts when operation==create and subresource=='' or operation==update and subresource==ephemeralcontainers"
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "ephemeralcontainers", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			EphemeralContainers: []api.EphemeralContainer{
				{
					EphemeralContainerCommon: api.EphemeralContainerCommon{
						Name: "container-2",
						EnvFrom: []api.EnvFromSource{{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
					},
				},
			},
		},
	}
	// validate enforces restrictions on secret mounts when operation==update and subresource==ephemeralcontainers"
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "ephemeralcontainers", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedSecretVolumes(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	// Add the default service account for the ns into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod1 := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod1, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}

	pod2 := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "container-1",
					Env: []api.EnvVar{
						{
							Name: "env-1",
							ValueFrom: &api.EnvVarSource{
								SecretKeyRef: &api.SecretKeySelector{
									LocalObjectReference: api.LocalObjectReference{Name: "foo"},
								},
							},
						},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envVar") {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "container-1",
					EnvFrom: []api.EnvFromSource{
						{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envFrom") {
		t.Errorf("Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			InitContainers: []api.Container{
				{
					Name: "container-1",
					Env: []api.EnvVar{
						{
							Name: "env-1",
							ValueFrom: &api.EnvVarSource{
								SecretKeyRef: &api.SecretKeySelector{
									LocalObjectReference: api.LocalObjectReference{Name: "foo"},
								},
							},
						},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("admit only enforces restrictions on secret mounts when operation==create. Unexpected error: %v", err)
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envVar") {
		t.Errorf("validate only enforces restrictions on secret mounts when operation==create and subresource==''. Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			InitContainers: []api.Container{
				{
					Name: "container-1",
					EnvFrom: []api.EnvFromSource{
						{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("admit only enforces restrictions on secret mounts when operation==create. Unexpected error: %v", err)
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envFrom") {
		t.Errorf("validate only enforces restrictions on secret mounts when operation==create and subresource==''. Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			EphemeralContainers: []api.EphemeralContainer{
				{
					EphemeralContainerCommon: api.EphemeralContainerCommon{
						Name: "container-2",
						Env: []api.EnvVar{
							{
								Name: "env-1",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{Name: "foo"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil); err != nil {
		t.Errorf("admit only enforces restrictions on secret mounts when operation==create and subresource==''. Unexpected error: %v", err)
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "ephemeralcontainers", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envVar") {
		t.Errorf("validate enforces restrictions on secret mounts when operation==update and subresource==ephemeralcontainers. Unexpected error: %v", err)
	}

	pod2 = &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: DefaultServiceAccountName,
			EphemeralContainers: []api.EphemeralContainer{
				{
					EphemeralContainerCommon: api.EphemeralContainerCommon{
						Name: "container-2",
						EnvFrom: []api.EnvFromSource{{
							SecretRef: &api.SecretEnvSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "foo"}}}},
					},
				},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "ephemeralcontainers", admission.Update, &metav1.UpdateOptions{}, false, nil)
	if err := admit.Validate(context.TODO(), attrs, nil); err == nil || !strings.Contains(err.Error(), "with envFrom") {
		t.Errorf("validate enforces restrictions on secret mounts when operation==update and subresource==ephemeralcontainers. Unexpected error: %v", err)
	}
}

func TestAllowUnreferencedSecretVolumesForPermissiveSAs(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = false

	// Add the default service account for the ns into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:        DefaultServiceAccountName,
			Namespace:   ns,
			Annotations: map[string]string{EnforceMountableSecretsAnnotation: "true"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestAllowsReferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []corev1.LocalObjectReference{
			{Name: "foo"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	// Add the default service account for the ns into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestDoNotAddImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []corev1.LocalObjectReference{
			{Name: "foo"},
			{Name: "bar"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(pod.Spec.ImagePullSecrets) != 1 || pod.Spec.ImagePullSecrets[0].Name != "foo" {
		t.Errorf("unexpected image pull secrets: %v", pod.Spec.ImagePullSecrets)
	}
}

func TestAddImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetExternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []corev1.LocalObjectReference{
			{Name: "foo"},
			{Name: "bar"},
		},
	}
	originalSA := sa.DeepCopy()
	expected := []api.LocalObjectReference{
		{Name: "foo"},
		{Name: "bar"},
	}
	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().V1().ServiceAccounts().Informer().GetStore().Add(sa)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	err := admissiontesting.WithReinvocationTesting(t, admit).Admit(context.TODO(), attrs, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	assert.EqualValues(t, expected, pod.Spec.ImagePullSecrets, "expected %v, got %v", expected, pod.Spec.ImagePullSecrets)

	pod.Spec.ImagePullSecrets[1] = api.LocalObjectReference{Name: "baz"}
	if !reflect.DeepEqual(originalSA, sa) {
		t.Errorf("accidentally mutated the ServiceAccount.ImagePullSecrets: %v", sa.ImagePullSecrets)
	}
}

func testGenerateName(n string) string {
	return n + "abc123"
}

var generatedVolumeName = testGenerateName(ServiceAccountVolumeName + "-")
