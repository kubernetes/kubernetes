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
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/kubelet/util/manager"
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
		return fakeClient.CoreV1().Secrets(namespace).Get(name, opts)
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
