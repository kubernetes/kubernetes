/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	kubelet "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/types"
)

func TestIgnoresNonCreate(t *testing.T) {
	pod := &api.Pod{}
	for _, op := range []admission.Operation{admission.Update, admission.Delete, admission.Connect} {
		attrs := admission.NewAttributesRecord(pod, "Pod", "myns", "myname", string(api.ResourcePods), "", op, nil)
		handler := admission.NewChainHandler(NewServiceAccount(nil))
		err := handler.Admit(attrs)
		if err != nil {
			t.Errorf("Expected %s operation allowed, got err: %v", op, err)
		}
	}
}

func TestIgnoresNonPodResource(t *testing.T) {
	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", "myns", "myname", "CustomResource", "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err != nil {
		t.Errorf("Expected non-pod resource allowed, got err: %v", err)
	}
}

func TestIgnoresNilObject(t *testing.T) {
	attrs := admission.NewAttributesRecord(nil, "Pod", "myns", "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err != nil {
		t.Errorf("Expected nil object allowed allowed, got err: %v", err)
	}
}

func TestIgnoresNonPodObject(t *testing.T) {
	obj := &api.Namespace{}
	attrs := admission.NewAttributesRecord(obj, "Pod", "myns", "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err != nil {
		t.Errorf("Expected non pod object allowed, got err: %v", err)
	}
}

func TestIgnoresMirrorPod(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", "myns", "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err != nil {
		t.Errorf("Expected mirror pod without service account or secrets allowed, got err: %v", err)
	}
}

func TestRejectsMirrorPodWithServiceAccount(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				kubelet.ConfigMirrorAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			ServiceAccountName: "default",
		},
	}
	attrs := admission.NewAttributesRecord(pod, "Pod", "myns", "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a service account")
	}
}

func TestRejectsMirrorPodWithSecretVolumes(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", "myns", "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := NewServiceAccount(nil).Admit(attrs)
	if err == nil {
		t.Errorf("Expected a mirror pod to be prevented from referencing a secret volume")
	}
}

func TestAssignsDefaultServiceAccountAndToleratesMissingAPIToken(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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

	admit := NewServiceAccount(nil)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected admission error for missing API token")
	}
}

func TestFetchesUncachedServiceAccount(t *testing.T) {
	ns := "myns"

	// Build a test client that the admission plugin can use to look up the service account missing from its cache
	client := testclient.NewSimpleFake(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	admit := NewServiceAccount(client)
	admit.RequireAPIToken = false

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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
	client := testclient.NewSimpleFake()

	admit := NewServiceAccount(client)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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

	admit := NewServiceAccount(nil)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns with a token into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
		Secrets: []api.ObjectReference{
			{Name: tokenName},
		},
	})
	// Add a token for the service account into the cache
	admit.secrets.Add(&api.Secret{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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

	admit := NewServiceAccount(nil)
	admit.MountServiceAccountToken = true
	admit.RequireAPIToken = true

	// Add the default service account for the ns with a token into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceAccountName,
			Namespace: ns,
			UID:       types.UID(serviceAccountUID),
		},
		Secrets: []api.ObjectReference{
			{Name: tokenName},
		},
	})
	// Add a token for the service account into the cache
	admit.secrets.Add(&api.Secret{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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
}

func TestAllowsReferencedSecretVolumes(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		Secrets: []api.ObjectReference{
			{Name: "foo"},
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedSecretVolumes(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}},
			},
		},
	}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestAllowsReferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestRejectsUnreferencedImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
	})

	pod := &api.Pod{
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{{Name: "foo"}},
		},
	}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
	err := admit.Admit(attrs)
	if err == nil {
		t.Errorf("Expected rejection for using a secret the service account does not reference")
	}
}

func TestDoNotAddImagePullSecrets(t *testing.T) {
	ns := "myns"

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	// Add the default service account for the ns with a secret reference into the cache
	admit.serviceAccounts.Add(&api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
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
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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

	admit := NewServiceAccount(nil)
	admit.LimitSecretReferences = true
	admit.RequireAPIToken = false

	sa := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      DefaultServiceAccountName,
			Namespace: ns,
		},
		ImagePullSecrets: []api.LocalObjectReference{
			{Name: "foo"},
			{Name: "bar"},
		},
	}
	// Add the default service account for the ns with a secret reference into the cache
	admit.serviceAccounts.Add(sa)

	pod := &api.Pod{}
	attrs := admission.NewAttributesRecord(pod, "Pod", ns, "myname", string(api.ResourcePods), "", admission.Create, nil)
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
