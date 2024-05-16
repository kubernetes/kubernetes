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
