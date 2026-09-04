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

package secret

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/kubelet/util/manager"
	"k8s.io/utils/clock"
)

func checkObject(t *testing.T, store manager.Store, ns, name string, shouldExist bool) {
	_, err := store.Get(ns, name)
	if shouldExist && err != nil {
		t.Errorf("unexpected actions: %#v", err)
	}
	if !shouldExist && (err == nil || !strings.Contains(err.Error(), fmt.Sprintf("object %q/%q not registered", ns, name))) {
		t.Errorf("unexpected actions: %#v", err)
	}
}

func noObjectTTL() (time.Duration, bool) {
	return time.Duration(0), false
}

func getSecret(fakeClient clientset.Interface) manager.GetObjectFunc {
	return func(namespace, name string, opts metav1.GetOptions) (runtime.Object, error) {
		return fakeClient.CoreV1().Secrets(namespace).Get(context.TODO(), name, opts)
	}
}

type envSecrets struct {
	envVarNames  []string
	envFromNames []string
}

type secretsToAttach struct {
	imagePullSecretNames []string
	containerEnvSecrets  []envSecrets
}

func podWithSecrets(ns, podName string, toAttach secretsToAttach) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      podName,
		},
		Spec: v1.PodSpec{},
	}
	for _, name := range toAttach.imagePullSecretNames {
		pod.Spec.ImagePullSecrets = append(
			pod.Spec.ImagePullSecrets, v1.LocalObjectReference{Name: name})
	}
	for i, secrets := range toAttach.containerEnvSecrets {
		container := v1.Container{
			Name: fmt.Sprintf("container-%d", i),
		}
		for _, name := range secrets.envFromNames {
			envFrom := v1.EnvFromSource{
				SecretRef: &v1.SecretEnvSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: name,
					},
				},
			}
			container.EnvFrom = append(container.EnvFrom, envFrom)
		}

		for _, name := range secrets.envVarNames {
			envSource := &v1.EnvVarSource{
				SecretKeyRef: &v1.SecretKeySelector{
					LocalObjectReference: v1.LocalObjectReference{
						Name: name,
					},
				},
			}
			container.Env = append(container.Env, v1.EnvVar{ValueFrom: envSource})
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}
	return pod
}

func TestCacheBasedSecretManager(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := manager.NewObjectStore(getSecret(fakeClient), clock.RealClock{}, noObjectTTL, 0)
	manager := &secretManager{
		manager: manager.NewCacheBasedManager(store, getSecretNames),
	}

	// Create a pod with some secrets.
	s1 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{
			{envVarNames: []string{"s1"}},
			{envVarNames: []string{"s2"}},
			{envFromNames: []string{"s20"}},
		},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s1))
	// Update the pod with a different secrets.
	s2 := secretsToAttach{
		imagePullSecretNames: []string{"s1"},
		containerEnvSecrets: []envSecrets{
			{envVarNames: []string{"s3"}},
			{envVarNames: []string{"s4"}},
			{envFromNames: []string{"s40"}},
		},
	}
	manager.RegisterPod(podWithSecrets("ns1", "name1", s2))
	// Create another pod, but with same secrets in different namespace.
	manager.RegisterPod(podWithSecrets("ns2", "name2", s2))
	// Create and delete a pod with some other secrets.
	s3 := secretsToAttach{
		imagePullSecretNames: []string{"s5"},
		containerEnvSecrets: []envSecrets{
			{envVarNames: []string{"s6"}},
			{envFromNames: []string{"s60"}},
		},
	}
	manager.RegisterPod(podWithSecrets("ns3", "name", s3))
	manager.UnregisterPod(podWithSecrets("ns3", "name", s3))

	// We should have only: s1, s3 and s4 secrets in namespaces: ns1 and ns2.
	for _, ns := range []string{"ns1", "ns2", "ns3"} {
		for _, secret := range []string{"s1", "s2", "s3", "s4", "s5", "s6", "s20", "s40", "s50"} {
			shouldExist :=
				(secret == "s1" || secret == "s3" || secret == "s4" || secret == "s40") && (ns == "ns1" || ns == "ns2")
			checkObject(t, store, ns, secret, shouldExist)
		}
	}
}

// stubObjectManager returns a fixed object and error, so the type assertion in
// secretManager.GetSecret can be exercised without building a real store.
type stubObjectManager struct {
	object runtime.Object
	err    error
}

func (s *stubObjectManager) GetObject(namespace, name string) (runtime.Object, error) {
	return s.object, s.err
}

func (s *stubObjectManager) RegisterPod(pod *v1.Pod) {}

func (s *stubObjectManager) UnregisterPod(pod *v1.Pod) {}

func TestSimpleSecretManager(t *testing.T) {
	secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "s1"}}
	secretManager := NewSimpleSecretManager(fake.NewSimpleClientset(secret))

	actual, err := secretManager.GetSecret("ns1", "s1")
	if err != nil {
		t.Fatalf("unexpected error getting ns1/s1: %v", err)
	}
	if actual.Namespace != "ns1" || actual.Name != "s1" {
		t.Errorf("expected ns1/s1, got %v/%v", actual.Namespace, actual.Name)
	}

	// Every call reads through to the apiserver, so a secret of the same name in
	// another namespace must not be visible.
	if _, err := secretManager.GetSecret("ns2", "s1"); !apierrors.IsNotFound(err) {
		t.Errorf("expected NotFound for ns2/s1, got %v", err)
	}

	// Register/UnregisterPod are no-ops for this implementation. Call them to
	// pin down that they do not panic and do not affect later reads.
	pod := podWithSecrets("ns1", "name1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	secretManager.RegisterPod(pod)
	secretManager.UnregisterPod(pod)
	if _, err := secretManager.GetSecret("ns1", "s1"); err != nil {
		t.Errorf("unexpected error after register/unregister: %v", err)
	}
}

func TestSecretManagerGetSecret(t *testing.T) {
	secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "s1"}}
	testCases := []struct {
		name         string
		object       runtime.Object
		err          error
		expectSecret *v1.Secret
		expectErr    string
	}{
		{
			name:         "secret is returned unchanged",
			object:       secret,
			expectSecret: secret,
		},
		{
			name:      "error from the store is propagated",
			err:       fmt.Errorf("object %q/%q not registered", "ns1", "s1"),
			expectErr: "not registered",
		},
		{
			name:      "object of another type is rejected",
			object:    &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "s1"}},
			expectErr: "unexpected object type",
		},
		{
			// A store returning no object and no error is not something callers
			// should silently treat as a missing secret.
			name:      "nil object without error is rejected",
			expectErr: "unexpected object type",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			secretManager := &secretManager{
				manager: &stubObjectManager{object: testCase.object, err: testCase.err},
			}

			actual, err := secretManager.GetSecret("ns1", "s1")
			if testCase.expectErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got secret %v", testCase.expectErr, actual)
				}
				if !strings.Contains(err.Error(), testCase.expectErr) {
					t.Errorf("expected error containing %q, got %v", testCase.expectErr, err)
				}
				if actual != nil {
					t.Errorf("expected no secret alongside the error, got %v", actual)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if actual != testCase.expectSecret {
				t.Errorf("expected secret %v, got %v", testCase.expectSecret, actual)
			}
		})
	}
}

func TestSecretManagerRegisterPodDelegates(t *testing.T) {
	stub := &stubObjectManager{object: &v1.Secret{}}
	secretManager := &secretManager{manager: stub}

	// Delegation only, so the assertion is simply that these reach the
	// underlying manager without panicking.
	pod := podWithSecrets("ns1", "name1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	secretManager.RegisterPod(pod)
	secretManager.UnregisterPod(pod)
}

func TestCachingAndWatchingSecretManagers(t *testing.T) {
	testCases := []struct {
		name          string
		secretManager Manager
	}{
		{
			name:          "caching",
			secretManager: NewCachingSecretManager(fake.NewSimpleClientset(), noObjectTTL),
		},
		{
			name:          "watching",
			secretManager: NewWatchingSecretManager(fake.NewSimpleClientset(), time.Minute),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			if testCase.secretManager == nil {
				t.Fatal("expected a secret manager")
			}

			// Nothing has been registered, so a lookup must fail locally rather
			// than fall back to the apiserver.
			if _, err := testCase.secretManager.GetSecret("ns1", "s1"); err == nil {
				t.Error("expected an error for a secret that was never registered")
			}
		})
	}
}

func TestFakeSecretManager(t *testing.T) {
	pod := podWithSecrets("ns1", "name1", secretsToAttach{imagePullSecretNames: []string{"s1"}})

	t.Run("without secrets", func(t *testing.T) {
		fakeManager := NewFakeManager()

		// Callers rely on the nil secret and nil error to skip secret handling
		// altogether, so both halves matter.
		actual, err := fakeManager.GetSecret("ns1", "s1")
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		if actual != nil {
			t.Errorf("expected no secret, got %v", actual)
		}

		fakeManager.RegisterPod(pod)
		fakeManager.UnregisterPod(pod)
	})

	t.Run("with secrets", func(t *testing.T) {
		secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "s1"}}
		fakeManager := NewFakeManagerWithSecrets([]*v1.Secret{secret})

		actual, err := fakeManager.GetSecret("ns1", "s1")
		if err != nil {
			t.Fatalf("unexpected error getting ns1/s1: %v", err)
		}
		if actual != secret {
			t.Errorf("expected secret %v, got %v", secret, actual)
		}

		// Lookup matches on name only, so the namespace argument is ignored.
		actual, err = fakeManager.GetSecret("other", "s1")
		if err != nil {
			t.Fatalf("unexpected error getting other/s1: %v", err)
		}
		if actual != secret {
			t.Errorf("expected secret %v, got %v", secret, actual)
		}

		if _, err := fakeManager.GetSecret("ns1", "missing"); err == nil {
			t.Error("expected an error for a secret that was not provided")
		}
	})
}
