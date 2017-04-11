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
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	"k8s.io/kubernetes/pkg/controller"
	kubelet "k8s.io/kubernetes/pkg/kubelet/types"
)

func TestIgnoresNonCreate(t *testing.T) {
	pod := &api.Pod{}
	for _, op := range []admission.Operation{admission.Update, admission.Delete, admission.Connect} {
		attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", op, nil)
		handler := admission.NewChainHandler(NewServiceAccount())
		err := handler.Admit(attrs)
		if err != nil {
			t.Errorf("Expected %s operation allowed, got err: %v", op, err)
		}
	}
}

func TestIgnoresNonPodResource(t *testing.T) {
	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("CustomResource").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
	if err != nil {
		t.Errorf("Expected non-pod resource allowed, got err: %v", err)
	}
}

func TestIgnoresNilObject(t *testing.T) {
	attrs := admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
	if err != nil {
		t.Errorf("Expected nil object allowed allowed, got err: %v", err)
	}
}

func TestIgnoresNonPodObject(t *testing.T) {
	obj := &api.Namespace{}
	attrs := admission.NewAttributesRecord(obj, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
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
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
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
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
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
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), "myns", "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := NewServiceAccount().Admit(attrs)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a secret volume")
	}
}

func TestAssignsDefaultServiceAccountAndToleratesMissingAPIToken(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if pod.Spec.ServiceAccountName != DefaultServiceAccountName {
		t.Errorf("Expected service account %s assigned, got %s", DefaultServiceAccountName, pod.Spec.ServiceAccountName)
	}
}

func TestAssignsDefaultServiceAccountAndRejectsMissingAPIToken(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil || !errors.IsServerTimeout(err) {
		t.Errorf("Expected server timeout error for missing API token: %v", err)
	}
}

func TestFetchesUncachedServiceAccount(t *testing.T) {
	ns := "myns"

	// Build a test client that the admission plugin can use to look up the service account missing from its cache
	client := fake.NewSimpleClientset(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.client = client
	admit.RequireAPIToken = false

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
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
	admit.SetInternalKubeClientSet(client)
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected error for missing service account, got none")
	}
}

func TestAutomountsAPIToken(t *testing.T) {
	ns := "myns"
	tokenName := "token-name"
	serviceAccountName := DefaultServiceAccountName
	serviceAccountUID := "12345"

	expectedVolume := api.Volume{
		Name: tokenName,
		VolumeSource: api.VolumeSource{
			Secret: &api.SecretVolumeSource{SecretName: tokenName},
		},
	}
	expectedVolumeMount := api.VolumeMount{
		Name:      tokenName,
		ReadOnly:  true,
		MountPath: DefaultAPITokenMountPath,
	}

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns with a token into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
		Secrets: []api.ObjectReference{
			{Name: tokenName},
		},
	})
	// Add a token for the service account into the cache
	informerFactory.Core().InternalVersion().Secrets().Informer().GetStore().Add(&api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      tokenName,
			Namespace: ns,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccountName,
				api.ServiceAccountUIDKey:  serviceAccountUID,
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			api.ServiceAccountTokenKey: []byte("token-data"),
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
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

	pod = &api.Pod{
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{},
			},
		},
	}
	attrs = admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
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
	tokenName := "token-name"
	serviceAccountName := DefaultServiceAccountName
	serviceAccountUID := "12345"

	expectedVolumeMount := api.VolumeMount{
		Name:      "my-custom-mount",
		ReadOnly:  false,
		MountPath: DefaultAPITokenMountPath,
	}

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns with a token into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
		Secrets: []api.ObjectReference{
			{Name: tokenName},
		},
	})
	// Add a token for the service account into the cache
	informerFactory.Core().InternalVersion().Secrets().Informer().GetStore().Add(&api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      tokenName,
			Namespace: ns,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccountName,
				api.ServiceAccountUIDKey:  serviceAccountUID,
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			api.ServiceAccountTokenKey: []byte("token-data"),
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
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
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
	attrs = admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
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
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		Secrets: []api.ObjectReference{
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
	attrs := admission.NewAttributesRecord(pod1, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
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
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
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
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedSecretVolumes(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
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
	attrs := admission.NewAttributesRecord(pod1, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err == nil {
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
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err == nil || !strings.Contains(err.Error(), "with envVar") {
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
	attrs = admission.NewAttributesRecord(pod2, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err == nil || !strings.Contains(err.Error(), "with envVar") {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestAllowUnreferencedSecretVolumesForPermissiveSAs(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = false
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
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
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestAllowsReferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []api.LocalObjectReference{
			{Name: "foo"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
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
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestDoNotAddImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(&api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []api.LocalObjectReference{
			{Name: "foo"},
			{Name: "bar"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
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
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []api.LocalObjectReference{
			{Name: "foo"},
			{Name: "bar"},
		},
	}
	// Add the default service account for the ns with a secret reference into the cache
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(sa)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(pod.Spec.ImagePullSecrets) != 2 || !reflect.DeepEqual(sa.ImagePullSecrets, pod.Spec.ImagePullSecrets) {
		t.Errorf("expected %v, got %v", sa.ImagePullSecrets, pod.Spec.ImagePullSecrets)
	}

	pod.Spec.ImagePullSecrets[1] = api.LocalObjectReference{Name: "baz"}
	if reflect.DeepEqual(sa.ImagePullSecrets, pod.Spec.ImagePullSecrets) {
		t.Errorf("accidentally mutated the ServiceAccount.ImagePullSecrets: %v", sa.ImagePullSecrets)
	}
}

func TestMultipleReferencedSecrets(t *testing.T) {
	var (
		ns                 = "myns"
		serviceAccountName = "mysa"
		serviceAccountUID  = "mysauid"
		token1             = "token1"
		token2             = "token2"
	)

	admit := NewServiceAccount()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	admit.SetInternalKubeInformerFactory(informerFactory)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccountName,
			UID:       types.UID(serviceAccountUID),
			Namespace: ns,
		},
		Secrets: []api.ObjectReference{
			{Name: token1},
			{Name: token2},
		},
	}
	informerFactory.Core().InternalVersion().ServiceAccounts().Informer().GetStore().Add(sa)

	// Add two tokens for the service account into the cache.
	informerFactory.Core().InternalVersion().Secrets().Informer().GetStore().Add(&api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      token2,
			Namespace: ns,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccountName,
				api.ServiceAccountUIDKey:  serviceAccountUID,
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			api.ServiceAccountTokenKey: []byte("token-data"),
		},
	})
	informerFactory.Core().InternalVersion().Secrets().Informer().GetStore().Add(&api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      token1,
			Namespace: ns,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccountName,
				api.ServiceAccountUIDKey:  serviceAccountUID,
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			api.ServiceAccountTokenKey: []byte("token-data"),
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ServiceAccountName: serviceAccountName,
			Containers: []api.Container{
				{Name: "container-1"},
			},
		},
	}

	attrs := admission.NewAttributesRecord(pod, nil, api.Kind("Pod").WithVersion("version"), ns, "myname", api.Resource("pods").WithVersion("version"), "", admission.Create, nil)
	if err := admit.Admit(attrs); err != nil {
		t.Fatal(err)
	}

	if n := len(pod.Spec.Volumes); n != 1 {
		t.Fatalf("expected 1 volume mount, got %d", n)
	}
	if name := pod.Spec.Volumes[0].Name; name != token1 {
		t.Errorf("expected first referenced secret to be mounted, got %q", name)
	}
}

func newSecret(secretType api.SecretType, namespace, name, serviceAccountName, serviceAccountUID string) *api.Secret {
	return &api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccountName,
				api.ServiceAccountUIDKey:  serviceAccountUID,
			},
		},
		Type: secretType,
	}
}

func TestGetServiceAccountTokens(t *testing.T) {
	admit := NewServiceAccount()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	admit.secretLister = corelisters.NewSecretLister(indexer)

	ns := "namespace"
	serviceAccountUID := "12345"

	sa := &api.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
	}

	nonSATokenSecret := newSecret(api.SecretTypeDockercfg, ns, "nonSATokenSecret", DefaultServiceAccountName, serviceAccountUID)
	indexer.Add(nonSATokenSecret)

	differentSAToken := newSecret(api.SecretTypeServiceAccountToken, ns, "differentSAToken", "someOtherSA", "someOtherUID")
	indexer.Add(differentSAToken)

	matchingSAToken := newSecret(api.SecretTypeServiceAccountToken, ns, "matchingSAToken", DefaultServiceAccountName, serviceAccountUID)
	indexer.Add(matchingSAToken)

	tokens, err := admit.getServiceAccountTokens(sa)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(tokens) != 1 {
		names := make([]string, 0, len(tokens))
		for _, token := range tokens {
			names = append(names, token.Name)
		}
		t.Fatalf("expected only 1 token, got %v", names)
	}
	if e, a := matchingSAToken.Name, tokens[0].Name; e != a {
		t.Errorf("expected token %s, got %s", e, a)
	}
}
