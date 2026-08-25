/*
Copyright 2026 The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
)

func newTestSecret(namespace, name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string][]byte{
			"key": []byte("value-" + name),
		},
	}
}

func TestSimpleSecretManager(t *testing.T) {
	fakeClient := fake.NewSimpleClientset(newTestSecret("ns1", "s1"))
	manager := NewSimpleSecretManager(fakeClient)

	secret, err := manager.GetSecret("ns1", "s1")
	if err != nil {
		t.Fatalf("unexpected error getting existing secret: %v", err)
	}
	if string(secret.Data["key"]) != "value-s1" {
		t.Errorf("unexpected secret data: %v", secret.Data)
	}

	if _, err := manager.GetSecret("ns1", "does-not-exist"); err == nil {
		t.Error("expected an error when getting a nonexistent secret")
	}
	if _, err := manager.GetSecret("ns2", "s1"); err == nil {
		t.Error("expected an error when getting a secret from a different namespace")
	}

	// Both operations are no-ops in the simple manager, but must not panic.
	pod := podWithSecrets("ns1", "pod1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	manager.RegisterPod(pod)
	manager.UnregisterPod(pod)
}

func TestCachingSecretManager(t *testing.T) {
	fakeClient := fake.NewSimpleClientset(newTestSecret("ns1", "s1"))
	manager := NewCachingSecretManager(fakeClient, noObjectTTL)

	pod := podWithSecrets("ns1", "pod1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	manager.RegisterPod(pod)

	secret, err := manager.GetSecret("ns1", "s1")
	if err != nil {
		t.Fatalf("unexpected error getting registered secret: %v", err)
	}
	if string(secret.Data["key"]) != "value-s1" {
		t.Errorf("unexpected secret data: %v", secret.Data)
	}

	// Getting a secret that is not referenced by any registered pod must fail.
	if _, err := manager.GetSecret("ns1", "not-registered"); err == nil {
		t.Error("expected an error when getting an unregistered secret")
	}

	manager.UnregisterPod(pod)
	_, err = manager.GetSecret("ns1", "s1")
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Errorf("expected a 'not registered' error after unregistering the pod, got: %v", err)
	}
}

func TestWatchingSecretManager(t *testing.T) {
	fakeClient := fake.NewSimpleClientset(newTestSecret("ns1", "s1"))
	manager := NewWatchingSecretManager(fakeClient, time.Minute)

	pod := podWithSecrets("ns1", "pod1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	manager.RegisterPod(pod)

	// The watch-based manager serves secrets from an informer cache, so wait
	// until the cache is populated.
	var secret *v1.Secret
	pollErr := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		var getErr error
		secret, getErr = manager.GetSecret("ns1", "s1")
		return getErr == nil, nil
	})
	if pollErr != nil {
		t.Fatalf("timed out waiting for the watching manager to serve the secret: %v", pollErr)
	}
	if string(secret.Data["key"]) != "value-s1" {
		t.Errorf("unexpected secret data: %v", secret.Data)
	}

	// Getting a secret that is not referenced by any registered pod must fail.
	if _, err := manager.GetSecret("ns1", "not-registered"); err == nil {
		t.Error("expected an error when getting an unregistered secret")
	}

	manager.UnregisterPod(pod)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		_, getErr := manager.GetSecret("ns1", "s1")
		return getErr != nil && strings.Contains(getErr.Error(), "not registered"), nil
	}); err != nil {
		t.Errorf("expected a 'not registered' error after unregistering the pod: %v", err)
	}
}

func TestFakeManager(t *testing.T) {
	manager := NewFakeManager()
	secret, err := manager.GetSecret("ns1", "s1")
	if secret != nil || err != nil {
		t.Errorf("expected a nil secret without error from an empty fake manager, got secret=%v, err=%v", secret, err)
	}

	fakeSecret := newTestSecret("ns1", "s1")
	manager = NewFakeManagerWithSecrets([]*v1.Secret{fakeSecret})
	secret, err = manager.GetSecret("ns1", "s1")
	if err != nil || secret != fakeSecret {
		t.Errorf("expected the seeded secret, got secret=%v, err=%v", secret, err)
	}
	if _, err := manager.GetSecret("ns1", "does-not-exist"); err == nil {
		t.Error("expected an error when getting a nonexistent secret from a seeded fake manager")
	}

	// Both operations are no-ops in the fake manager, but must not panic.
	pod := podWithSecrets("ns1", "pod1", secretsToAttach{imagePullSecretNames: []string{"s1"}})
	manager.RegisterPod(pod)
	manager.UnregisterPod(pod)
}

// stubObjectManager is a manager.Manager whose GetObject returns a
// non-Secret object, to exercise the type assertion error path of
// secretManager.GetSecret.
type stubObjectManager struct{}

func (s *stubObjectManager) GetObject(namespace, name string) (runtime.Object, error) {
	return &v1.ConfigMap{}, nil
}

func (s *stubObjectManager) RegisterPod(pod *v1.Pod) {}

func (s *stubObjectManager) UnregisterPod(pod *v1.Pod) {}

func TestSecretManagerGetSecretUnexpectedObjectType(t *testing.T) {
	manager := &secretManager{manager: &stubObjectManager{}}
	if _, err := manager.GetSecret("ns1", "s1"); err == nil || !strings.Contains(err.Error(), "unexpected object type") {
		t.Errorf("expected an 'unexpected object type' error, got: %v", err)
	}
}
