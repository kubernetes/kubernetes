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
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	core "k8s.io/client-go/testing"

	"github.com/stretchr/testify/assert"
)

func checkConfigMap(t *testing.T, store *configMapStore, ns, name string, shouldExist bool) {
	_, err := store.Get(ns, name)
	if shouldExist && err != nil {
		t.Errorf("unexpected actions: %#v", err)
	}
	if !shouldExist && (err == nil || !strings.Contains(err.Error(), fmt.Sprintf("configmap %q/%q not registered", ns, name))) {
		t.Errorf("unexpected actions: %#v", err)
	}
}

func noObjectTTL() (time.Duration, bool) {
	return time.Duration(0), false
}

func TestConfigMapStore(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := newConfigMapStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	store.Add("ns1", "name1")
	store.Add("ns2", "name2")
	store.Add("ns1", "name1")
	store.Add("ns1", "name1")
	store.Delete("ns1", "name1")
	store.Delete("ns2", "name2")
	store.Add("ns3", "name3")

	// Adds don't issue Get requests.
	actions := fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	// Should issue Get request
	store.Get("ns1", "name1")
	// Shouldn't issue Get request, as configMap is not registered
	store.Get("ns2", "name2")
	// Should issue Get request
	store.Get("ns3", "name3")

	actions = fakeClient.Actions()
	assert.Equal(t, 2, len(actions), "unexpected actions: %#v", actions)

	for _, a := range actions {
		assert.True(t, a.Matches("get", "configmaps"), "unexpected actions: %#v", a)
	}

	checkConfigMap(t, store, "ns1", "name1", true)
	checkConfigMap(t, store, "ns2", "name2", false)
	checkConfigMap(t, store, "ns3", "name3", true)
	checkConfigMap(t, store, "ns4", "name4", false)
}

func TestConfigMapStoreDeletingConfigMap(t *testing.T) {
	fakeClient := &fake.Clientset{}
	store := newConfigMapStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	store.Add("ns", "name")

	result := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "name", ResourceVersion: "10"}}
	fakeClient.AddReactor("get", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		return true, result, nil
	})
	configMap, err := store.Get("ns", "name")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(configMap, result) {
		t.Errorf("Unexpected configMap: %v", configMap)
	}

	fakeClient.PrependReactor("get", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.ConfigMap{}, apierrors.NewNotFound(v1.Resource("configMap"), "name")
	})
	configMap, err = store.Get("ns", "name")
	if err == nil || !apierrors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(configMap, &v1.ConfigMap{}) {
		t.Errorf("Unexpected configMap: %v", configMap)
	}
}

func TestConfigMapStoreGetAlwaysRefresh(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newConfigMapStore(fakeClient, fakeClock, noObjectTTL, 0)

	for i := 0; i < 10; i++ {
		store.Add(fmt.Sprintf("ns-%d", i), fmt.Sprintf("name-%d", i))
	}
	fakeClient.ClearActions()

	wg := sync.WaitGroup{}
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func(i int) {
			store.Get(fmt.Sprintf("ns-%d", i%10), fmt.Sprintf("name-%d", i%10))
			wg.Done()
		}(i)
	}
	wg.Wait()
	actions := fakeClient.Actions()
	assert.Equal(t, 100, len(actions), "unexpected actions: %#v", actions)

	for _, a := range actions {
		assert.True(t, a.Matches("get", "configmaps"), "unexpected actions: %#v", a)
	}
}

func TestConfigMapStoreGetNeverRefresh(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newConfigMapStore(fakeClient, fakeClock, noObjectTTL, time.Minute)

	for i := 0; i < 10; i++ {
		store.Add(fmt.Sprintf("ns-%d", i), fmt.Sprintf("name-%d", i))
	}
	fakeClient.ClearActions()

	wg := sync.WaitGroup{}
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func(i int) {
			store.Get(fmt.Sprintf("ns-%d", i%10), fmt.Sprintf("name-%d", i%10))
			wg.Done()
		}(i)
	}
	wg.Wait()
	actions := fakeClient.Actions()
	// Only first Get, should forward the Get request.
	assert.Equal(t, 10, len(actions), "unexpected actions: %#v", actions)
}

func TestCustomTTL(t *testing.T) {
	ttl := time.Duration(0)
	ttlExists := false
	customTTL := func() (time.Duration, bool) {
		return ttl, ttlExists
	}

	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Time{})
	store := newConfigMapStore(fakeClient, fakeClock, customTTL, time.Minute)

	store.Add("ns", "name")
	store.Get("ns", "name")
	fakeClient.ClearActions()

	// Set 0-ttl and see if that works.
	ttl = time.Duration(0)
	ttlExists = true
	store.Get("ns", "name")
	actions := fakeClient.Actions()
	assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()

	// Set 5-minute ttl and see if this works.
	ttl = time.Duration(5) * time.Minute
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	// Still no effect after 4 minutes.
	fakeClock.Step(4 * time.Minute)
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	// Now it should have an effect.
	fakeClock.Step(time.Minute)
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()

	// Now remove the custom ttl and see if that works.
	ttlExists = false
	fakeClock.Step(55 * time.Second)
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 0, len(actions), "unexpected actions: %#v", actions)
	// Pass the minute and it should be triggered now.
	fakeClock.Step(5 * time.Second)
	store.Get("ns", "name")
	actions = fakeClient.Actions()
	assert.Equal(t, 1, len(actions), "unexpected actions: %#v", actions)
}

func TestParseNodeAnnotation(t *testing.T) {
	testCases := []struct {
		node   *v1.Node
		err    error
		exists bool
		ttl    time.Duration
	}{
		{
			node:   nil,
			err:    fmt.Errorf("error"),
			exists: false,
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node",
				},
			},
			exists: false,
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node",
					Annotations: map[string]string{},
				},
			},
			exists: false,
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node",
					Annotations: map[string]string{v1.ObjectTTLAnnotationKey: "bad"},
				},
			},
			exists: false,
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node",
					Annotations: map[string]string{v1.ObjectTTLAnnotationKey: "0"},
				},
			},
			exists: true,
			ttl:    time.Duration(0),
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node",
					Annotations: map[string]string{v1.ObjectTTLAnnotationKey: "60"},
				},
			},
			exists: true,
			ttl:    time.Minute,
		},
	}
	for i, testCase := range testCases {
		getNode := func() (*v1.Node, error) { return testCase.node, testCase.err }
		ttl, exists := GetObjectTTLFromNodeFunc(getNode)()
		if exists != testCase.exists {
			t.Errorf("%d: incorrect parsing: %t", i, exists)
			continue
		}
		if exists && ttl != testCase.ttl {
			t.Errorf("%d: incorrect ttl: %v", i, ttl)
		}
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

func podWithConfigMaps(ns, name string, toAttach configMapsToAttach) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
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

func TestCacheInvalidation(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newConfigMapStore(fakeClient, fakeClock, noObjectTTL, time.Minute)
	manager := &cachingConfigMapManager{
		configMapStore: store,
		registeredPods: make(map[objectKey]*v1.Pod),
	}

	// Create a pod with some configMaps.
	s1 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s1"}, envFromNames: []string{"s10"}},
			{envVarNames: []string{"s2"}},
		},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name1", s1))
	// Fetch both configMaps - this should triggger get operations.
	store.Get("ns1", "s1")
	store.Get("ns1", "s10")
	store.Get("ns1", "s2")
	actions := fakeClient.Actions()
	assert.Equal(t, 3, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()

	// Update a pod with a new configMap.
	s2 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s1"}},
			{envVarNames: []string{"s2"}, envFromNames: []string{"s20"}},
		},
		volumes: []string{"s3"},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name1", s2))
	// All configMaps should be invalidated - this should trigger get operations.
	store.Get("ns1", "s1")
	store.Get("ns1", "s2")
	store.Get("ns1", "s20")
	store.Get("ns1", "s3")
	actions = fakeClient.Actions()
	assert.Equal(t, 4, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()

	// Create a new pod that is refencing the first three configMaps - those should
	// be invalidated.
	manager.RegisterPod(podWithConfigMaps("ns1", "name2", s1))
	store.Get("ns1", "s1")
	store.Get("ns1", "s10")
	store.Get("ns1", "s2")
	store.Get("ns1", "s20")
	store.Get("ns1", "s3")
	actions = fakeClient.Actions()
	assert.Equal(t, 3, len(actions), "unexpected actions: %#v", actions)
	fakeClient.ClearActions()
}

func TestCacheRefcounts(t *testing.T) {
	fakeClient := &fake.Clientset{}
	fakeClock := clock.NewFakeClock(time.Now())
	store := newConfigMapStore(fakeClient, fakeClock, noObjectTTL, time.Minute)
	manager := &cachingConfigMapManager{
		configMapStore: store,
		registeredPods: make(map[objectKey]*v1.Pod),
	}

	s1 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s1"}, envFromNames: []string{"s10"}},
			{envVarNames: []string{"s2"}},
		},
		volumes: []string{"s3"},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name1", s1))
	manager.RegisterPod(podWithConfigMaps("ns1", "name2", s1))
	s2 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s4"}},
			{envVarNames: []string{"s5"}, envFromNames: []string{"s50"}},
		},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name2", s2))
	manager.RegisterPod(podWithConfigMaps("ns1", "name3", s2))
	manager.RegisterPod(podWithConfigMaps("ns1", "name4", s2))
	manager.UnregisterPod(podWithConfigMaps("ns1", "name3", s2))
	s3 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s3"}, envFromNames: []string{"s30"}},
			{envVarNames: []string{"s5"}},
		},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name5", s3))
	manager.RegisterPod(podWithConfigMaps("ns1", "name6", s3))
	s4 := configMapsToAttach{
		containerEnvConfigMaps: []envConfigMaps{
			{envVarNames: []string{"s6"}},
			{envFromNames: []string{"s60"}},
		},
	}
	manager.RegisterPod(podWithConfigMaps("ns1", "name7", s4))
	manager.UnregisterPod(podWithConfigMaps("ns1", "name7", s4))

	// Also check the Add + Update + Remove scenario.
	manager.RegisterPod(podWithConfigMaps("ns1", "other-name", s1))
	manager.RegisterPod(podWithConfigMaps("ns1", "other-name", s2))
	manager.UnregisterPod(podWithConfigMaps("ns1", "other-name", s2))

	refs := func(ns, name string) int {
		store.lock.Lock()
		defer store.lock.Unlock()
		item, ok := store.items[objectKey{ns, name}]
		if !ok {
			return 0
		}
		return item.refCount
	}
	assert.Equal(t, refs("ns1", "s1"), 1)
	assert.Equal(t, refs("ns1", "s10"), 1)
	assert.Equal(t, refs("ns1", "s2"), 1)
	assert.Equal(t, refs("ns1", "s3"), 3)
	assert.Equal(t, refs("ns1", "s30"), 2)
	assert.Equal(t, refs("ns1", "s4"), 2)
	assert.Equal(t, refs("ns1", "s5"), 4)
	assert.Equal(t, refs("ns1", "s50"), 2)
	assert.Equal(t, refs("ns1", "s6"), 0)
	assert.Equal(t, refs("ns1", "s60"), 0)
	assert.Equal(t, refs("ns1", "s7"), 0)
}

func TestCachingConfigMapManager(t *testing.T) {
	fakeClient := &fake.Clientset{}
	configMapStore := newConfigMapStore(fakeClient, clock.RealClock{}, noObjectTTL, 0)
	manager := &cachingConfigMapManager{
		configMapStore: configMapStore,
		registeredPods: make(map[objectKey]*v1.Pod),
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
			checkConfigMap(t, configMapStore, ns, configMap, shouldExist(ns, configMap))
		}
	}
}
