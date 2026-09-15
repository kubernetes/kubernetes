/*
Copyright 2017 The Kubernetes Authors.

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

package configmap

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

func getConfigMap(fakeClient clientset.Interface) manager.GetObjectFunc {
	return func(namespace, name string, opts metav1.GetOptions) (runtime.Object, error) {
		return fakeClient.CoreV1().ConfigMaps(namespace).Get(context.TODO(), name, opts)
	}
}

type envConfigMaps struct {
	envVarNames  []string
	envFromNames []string
}

type configMapsToAttach struct {
	containerEnvConfigMaps []envConfigMaps
	volumes                []string
}

func podWithConfigMaps(ns, podName string, toAttach configMapsToAttach) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      podName,
		},
		Spec: v1.PodSpec{},
	}
	for i, configMaps := range toAttach.containerEnvConfigMaps {
		container := v1.Container{
			Name: fmt.Sprintf("container-%d", i),
		}
		for _, name := range configMaps.envFromNames {
			envFrom := v1.EnvFromSource{
				ConfigMapRef: &v1.ConfigMapEnvSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: name,
					},
				},
			}
			container.EnvFrom = append(container.EnvFrom, envFrom)
		}

		for _, name := range configMaps.envVarNames {
			envSource := &v1.EnvVarSource{
				ConfigMapKeyRef: &v1.ConfigMapKeySelector{
					LocalObjectReference: v1.LocalObjectReference{
						Name: name,
					},
				},
			}
			container.Env = append(container.Env, v1.EnvVar{ValueFrom: envSource})
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}
	for _, configMap := range toAttach.volumes {
		volume := &v1.ConfigMapVolumeSource{
			LocalObjectReference: v1.LocalObjectReference{Name: configMap},
		}
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: configMap,
			VolumeSource: v1.VolumeSource{
				ConfigMap: volume,
			},
		})
	}
	return pod
}

func TestCacheBasedConfigMapManager(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := manager.NewObjectStore(getConfigMap(fakeClient), clock.RealClock{}, noObjectTTL, 0)
	manager := &configMapManager{
		manager: manager.NewCacheBasedManager(store, getConfigMapNames),
	}

	// Create a pod with some configMaps.
	s1 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s1"}},
			{envFromNames: []string{"s20"}},
		},
		volumes: []string{"s2"},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name1", s1))
	manager.RegisterPod(podWithConfigMaps("ns2", "name2", s1))
	// Update the pod with a different configMaps.
	s2 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s3"}},
			{envVarNames: []string{"s4"}},
			{envFromNames: []string{"s40"}},
		},
	}
	// Create another pod, but with same configMaps in different namespace.
	manager.RegisterPod(podWithConfigMaps("ns2", "name2", s2))
	// Create and delete a pod with some other configMaps.
	s3 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s6"}},
			{envFromNames: []string{"s60"}},
		},
	}
	manager.RegisterPod(podWithConfigMaps("ns3", "name", s3))
	manager.UnregisterPod(podWithConfigMaps("ns3", "name", s3))

	existingMaps := map[string][]string{
		"ns1": {"s1", "s2", "s20"},
		"ns2": {"s3", "s4", "s40"},
	}
	shouldExist := func(ns, configMap string) bool {
		if cmaps, ok := existingMaps[ns]; ok {
			for _, cm := range cmaps {
				if cm == configMap {
					return true
				}
			}
		}
		return false
	}

	for _, ns := range []string{"ns1", "ns2", "ns3"} {
		for _, configMap := range []string{"s1", "s2", "s3", "s4", "s5", "s6", "s20", "s40", "s50"} {
			checkObject(t, store, ns, configMap, shouldExist(ns, configMap))
		}
	}
}

// stubObjectManager returns a fixed object and error, so the type assertion in
// configMapManager.GetConfigMap can be exercised without building a real store.
type stubObjectManager struct {
	object runtime.Object
	err    error
}

func (s *stubObjectManager) GetObject(namespace, name string) (runtime.Object, error) {
	return s.object, s.err
}

func (s *stubObjectManager) RegisterPod(pod *v1.Pod) {}

func (s *stubObjectManager) UnregisterPod(pod *v1.Pod) {}

func TestSimpleConfigMapManager(t *testing.T) {
	configMap := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cm1"}}
	configMapManager := NewSimpleConfigMapManager(fake.NewSimpleClientset(configMap))

	actual, err := configMapManager.GetConfigMap("ns1", "cm1")
	if err != nil {
		t.Fatalf("unexpected error getting ns1/cm1: %v", err)
	}
	if actual.Namespace != "ns1" || actual.Name != "cm1" {
		t.Errorf("expected ns1/cm1, got %v/%v", actual.Namespace, actual.Name)
	}

	// Every call reads through to the apiserver, so a configmap of the same name
	// in another namespace must not be visible.
	if _, err := configMapManager.GetConfigMap("ns2", "cm1"); !apierrors.IsNotFound(err) {
		t.Errorf("expected NotFound for ns2/cm1, got %v", err)
	}

	// Register/UnregisterPod are no-ops for this implementation. Call them to
	// pin down that they do not panic and do not affect later reads.
	pod := podWithConfigMaps("ns1", "name1", configMapsToAttach{volumes: []string{"cm1"}})
	configMapManager.RegisterPod(pod)
	configMapManager.UnregisterPod(pod)
	if _, err := configMapManager.GetConfigMap("ns1", "cm1"); err != nil {
		t.Errorf("unexpected error after register/unregister: %v", err)
	}
}

func TestConfigMapManagerGetConfigMap(t *testing.T) {
	configMap := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cm1"}}
	testCases := []struct {
		name            string
		object          runtime.Object
		err             error
		expectConfigMap *v1.ConfigMap
		expectErr       string
	}{
		{
			name:            "configmap is returned unchanged",
			object:          configMap,
			expectConfigMap: configMap,
		},
		{
			name:      "error from the store is propagated",
			err:       fmt.Errorf("object %q/%q not registered", "ns1", "cm1"),
			expectErr: "not registered",
		},
		{
			name:      "object of another type is rejected",
			object:    &v1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cm1"}},
			expectErr: "unexpected object type",
		},
		{
			// A store returning no object and no error is not something callers
			// should silently treat as a missing configmap.
			name:      "nil object without error is rejected",
			expectErr: "unexpected object type",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			configMapManager := &configMapManager{
				manager: &stubObjectManager{object: testCase.object, err: testCase.err},
			}

			actual, err := configMapManager.GetConfigMap("ns1", "cm1")
			if testCase.expectErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got configmap %v", testCase.expectErr, actual)
				}
				if !strings.Contains(err.Error(), testCase.expectErr) {
					t.Errorf("expected error containing %q, got %v", testCase.expectErr, err)
				}
				if actual != nil {
					t.Errorf("expected no configmap alongside the error, got %v", actual)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if actual != testCase.expectConfigMap {
				t.Errorf("expected configmap %v, got %v", testCase.expectConfigMap, actual)
			}
		})
	}
}

func TestConfigMapManagerRegisterPodDelegates(t *testing.T) {
	stub := &stubObjectManager{object: &v1.ConfigMap{}}
	configMapManager := &configMapManager{manager: stub}

	// Delegation only, so the assertion is simply that these reach the
	// underlying manager without panicking.
	pod := podWithConfigMaps("ns1", "name1", configMapsToAttach{volumes: []string{"cm1"}})
	configMapManager.RegisterPod(pod)
	configMapManager.UnregisterPod(pod)
}

func TestCachingAndWatchingConfigMapManagers(t *testing.T) {
	testCases := []struct {
		name             string
		configMapManager Manager
	}{
		{
			name:             "caching",
			configMapManager: NewCachingConfigMapManager(fake.NewSimpleClientset(), noObjectTTL),
		},
		{
			name:             "watching",
			configMapManager: NewWatchingConfigMapManager(fake.NewSimpleClientset(), time.Minute),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			if testCase.configMapManager == nil {
				t.Fatal("expected a configmap manager")
			}

			// Nothing has been registered, so a lookup must fail locally rather
			// than fall back to the apiserver.
			if _, err := testCase.configMapManager.GetConfigMap("ns1", "cm1"); err == nil {
				t.Error("expected an error for a configmap that was never registered")
			}
		})
	}
}

func TestFakeConfigMapManager(t *testing.T) {
	fakeManager := NewFakeManager()

	// Callers rely on the nil configmap and nil error to skip configmap handling
	// altogether, so both halves matter.
	actual, err := fakeManager.GetConfigMap("ns1", "cm1")
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if actual != nil {
		t.Errorf("expected no configmap, got %v", actual)
	}

	pod := podWithConfigMaps("ns1", "name1", configMapsToAttach{volumes: []string{"cm1"}})
	fakeManager.RegisterPod(pod)
	fakeManager.UnregisterPod(pod)
}
