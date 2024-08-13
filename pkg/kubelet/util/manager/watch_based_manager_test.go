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

package manager

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"

	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"

	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"

	"github.com/stretchr/testify/assert"
)

func listSecret(fakeClient clientset.Interface) listObjectFunc {
	return func(namespace string, opts metav1.ListOptions) (runtime.Object, error) {
		return fakeClient.CoreV1().Secrets(namespace).List(context.TODO(), opts)
	}
}

func watchSecret(fakeClient clientset.Interface) watchObjectFunc {
	return func(namespace string, opts metav1.ListOptions) (watch.Interface, error) {
		return fakeClient.CoreV1().Secrets(namespace).Watch(context.TODO(), opts)
	}
}

func isSecretImmutable(object runtime.Object) bool {
	if secret, ok := object.(*v1.Secret); ok {
		return secret.Immutable != nil && *secret.Immutable
	}
	return false
}

func newSecretCache(fakeClient clientset.Interface, fakeClock clock.Clock, maxIdleTime time.Duration) *objectCache {
	return &objectCache{
		listObject:    listSecret(fakeClient),
		watchObject:   watchSecret(fakeClient),
		newObject:     func() runtime.Object { return &v1.Secret{} },
		isImmutable:   isSecretImmutable,
		groupResource: corev1.Resource("secret"),
		clock:         fakeClock,
		maxIdleTime:   maxIdleTime,
		items:         make(map[objectKey]*objectCacheItem),
	}
}

func TestSecretCache(t *testing.T) {
	fakeClient := &fake.Clientset{}

	listReactor := func(a core.Action) (bool, runtime.Object, error) {
		result := &v1.SecretList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: "123",
			},
		}
		return true, result, nil
	}
	fakeClient.AddReactor("list", "secrets", listReactor)
	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))

	fakeClock := testingclock.NewFakeClock(time.Now())
	store := newSecretCache(fakeClient, fakeClock, time.Minute)

	store.AddReference("ns", "name", "pod")
	_, err := store.Get("ns", "name")
	if !apierrors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got: %v", err)
	}

	// Eventually we should be able to read added secret.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "name", Namespace: "ns", ResourceVersion: "125"},
	}
	fakeWatch.Add(secret)
	getFn := func() (bool, error) {
		object, err := store.Get("ns", "name")
		if err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		secret := object.(*v1.Secret)
		if secret == nil || secret.Name != "name" || secret.Namespace != "ns" {
			return false, fmt.Errorf("unexpected secret: %v", secret)
		}
		return true, nil
	}
	if err := wait.PollImmediate(10*time.Millisecond, time.Second, getFn); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Eventually we should observer secret deletion.
	fakeWatch.Delete(secret)
	getFn = func() (bool, error) {
		_, err := store.Get("ns", "name")
		if err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	}
	if err := wait.PollImmediate(10*time.Millisecond, time.Second, getFn); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	store.DeleteReference("ns", "name", "pod")
	_, err = store.Get("ns", "name")
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSecretCacheMultipleRegistrations(t *testing.T) {
	fakeClient := &fake.Clientset{}

	listReactor := func(a core.Action) (bool, runtime.Object, error) {
		result := &v1.SecretList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: "123",
			},
		}
		return true, result, nil
	}
	fakeClient.AddReactor("list", "secrets", listReactor)
	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))

	fakeClock := testingclock.NewFakeClock(time.Now())
	store := newSecretCache(fakeClient, fakeClock, time.Minute)

	store.AddReference("ns", "name", "pod")
	// This should trigger List and Watch actions eventually.
	actionsFn := func() (bool, error) {
		actions := fakeClient.Actions()
		if len(actions) > 2 {
			return false, fmt.Errorf("too many actions: %v", actions)
		}
		if len(actions) < 2 {
			return false, nil
		}
		if actions[0].GetVerb() != "list" || actions[1].GetVerb() != "watch" {
			return false, fmt.Errorf("unexpected actions: %v", actions)
		}
		return true, nil
	}
	if err := wait.PollImmediate(10*time.Millisecond, time.Second, actionsFn); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Next registrations shouldn't trigger any new actions.
	for i := 0; i < 20; i++ {
		store.AddReference("ns", "name", types.UID(fmt.Sprintf("pod-%d", i)))
		store.DeleteReference("ns", "name", types.UID(fmt.Sprintf("pod-%d", i)))
	}
	actions := fakeClient.Actions()
	assert.Len(t, actions, 2, "unexpected actions")

	// Final delete also doesn't trigger any action.
	store.DeleteReference("ns", "name", "pod")
	_, err := store.Get("ns", "name")
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Errorf("unexpected error: %v", err)
	}
	actions = fakeClient.Actions()
	assert.Len(t, actions, 2, "unexpected actions")
}

func TestImmutableSecretStopsTheReflector(t *testing.T) {
	secret := func(rv string, immutable bool) *v1.Secret {
		result := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "name",
				Namespace:       "ns",
				ResourceVersion: rv,
			},
		}
		if immutable {
			trueVal := true
			result.Immutable = &trueVal
		}
		return result
	}

	tests := []struct {
		desc     string
		initial  *v1.Secret
		eventual *v1.Secret
	}{
		{
			desc:     "secret doesn't exist, created as mutable",
			initial:  nil,
			eventual: secret("200", false),
		},
		{
			desc:     "secret doesn't exist, created as immutable",
			initial:  nil,
			eventual: secret("200", true),
		},
		{
			desc:     "mutable secret modified to mutable",
			initial:  secret("100", false),
			eventual: secret("200", false),
		},
		{
			desc:     "mutable secret modified to immutable",
			initial:  secret("100", false),
			eventual: secret("200", true),
		},
		{
			desc:     "immutable secret",
			initial:  secret("100", true),
			eventual: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			fakeClient := &fake.Clientset{}
			listReactor := func(a core.Action) (bool, runtime.Object, error) {
				result := &v1.SecretList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "100",
					},
				}
				if tc.initial != nil {
					result.Items = []v1.Secret{*tc.initial}
				}
				return true, result, nil
			}
			fakeClient.AddReactor("list", "secrets", listReactor)
			fakeWatch := watch.NewFake()
			fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))

			fakeClock := testingclock.NewFakeClock(time.Now())
			store := newSecretCache(fakeClient, fakeClock, time.Minute)

			key := objectKey{namespace: "ns", name: "name"}
			itemExists := func() (bool, error) {
				store.lock.Lock()
				defer store.lock.Unlock()
				_, ok := store.items[key]
				return ok, nil
			}
			reflectorRunning := func() bool {
				store.lock.Lock()
				defer store.lock.Unlock()
				item := store.items[key]

				item.lock.Lock()
				defer item.lock.Unlock()
				return !item.stopped
			}

			// AddReference should start reflector.
			store.AddReference("ns", "name", "pod")
			if err := wait.Poll(10*time.Millisecond, time.Second, itemExists); err != nil {
				t.Errorf("item wasn't added to cache")
			}

			obj, err := store.Get("ns", "name")
			if tc.initial != nil {
				assert.True(t, apiequality.Semantic.DeepEqual(tc.initial, obj))
			} else {
				assert.True(t, apierrors.IsNotFound(err))
			}

			// Reflector should already be stopped for immutable secrets.
			assert.Equal(t, tc.initial == nil || !isSecretImmutable(tc.initial), reflectorRunning())

			if tc.eventual == nil {
				return
			}
			fakeWatch.Add(tc.eventual)

			// Eventually Get should return that secret.
			getFn := func() (bool, error) {
				object, err := store.Get("ns", "name")
				if err != nil {
					if apierrors.IsNotFound(err) {
						return false, nil
					}
					return false, err
				}
				secret := object.(*v1.Secret)
				return apiequality.Semantic.DeepEqual(tc.eventual, secret), nil
			}
			if err := wait.PollImmediate(10*time.Millisecond, time.Second, getFn); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// Reflector should already be stopped for immutable secrets.
			assert.Equal(t, tc.eventual == nil || !isSecretImmutable(tc.eventual), reflectorRunning())
		})
	}
}

func TestMaxIdleTimeStopsTheReflector(t *testing.T) {
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "name",
			Namespace:       "ns",
			ResourceVersion: "200",
		},
	}

	fakeClient := &fake.Clientset{}
	listReactor := func(a core.Action) (bool, runtime.Object, error) {
		result := &v1.SecretList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: "200",
			},
			Items: []v1.Secret{*secret},
		}

		return true, result, nil
	}

	fakeClient.AddReactor("list", "secrets", listReactor)
	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))
	fakeClock := testingclock.NewFakeClock(time.Now())
	store := newSecretCache(fakeClient, fakeClock, time.Minute)

	key := objectKey{namespace: "ns", name: "name"}
	itemExists := func() (bool, error) {
		store.lock.Lock()
		defer store.lock.Unlock()
		_, ok := store.items[key]
		return ok, nil
	}

	reflectorRunning := func() bool {
		store.lock.Lock()
		defer store.lock.Unlock()
		item := store.items[key]

		item.lock.Lock()
		defer item.lock.Unlock()
		return !item.stopped
	}

	// AddReference should start reflector.
	store.AddReference("ns", "name", "pod")
	if err := wait.Poll(10*time.Millisecond, 10*time.Second, itemExists); err != nil {
		t.Errorf("item wasn't added to cache")
	}

	obj, _ := store.Get("ns", "name")
	assert.True(t, apiequality.Semantic.DeepEqual(secret, obj))

	assert.True(t, reflectorRunning())

	fakeClock.Step(90 * time.Second)
	store.startRecycleIdleWatch()

	// Reflector should already be stopped for maxIdleTime exceeded.
	assert.False(t, reflectorRunning())

	obj, _ = store.Get("ns", "name")
	assert.True(t, apiequality.Semantic.DeepEqual(secret, obj))
	// Reflector should reRun after get secret again.
	assert.True(t, reflectorRunning())

	fakeClock.Step(20 * time.Second)
	_, _ = store.Get("ns", "name")
	fakeClock.Step(20 * time.Second)
	_, _ = store.Get("ns", "name")
	fakeClock.Step(20 * time.Second)
	_, _ = store.Get("ns", "name")
	store.startRecycleIdleWatch()

	// Reflector should be running when the get function is called periodically.
	assert.True(t, reflectorRunning())
}

func TestReflectorNotStoppedOnSlowInitialization(t *testing.T) {
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "name",
			Namespace:       "ns",
			ResourceVersion: "200",
		},
	}

	fakeClock := testingclock.NewFakeClock(time.Now())

	fakeClient := &fake.Clientset{}
	listReactor := func(a core.Action) (bool, runtime.Object, error) {
		<-fakeClock.After(120 * time.Second)

		result := &v1.SecretList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: "200",
			},
			Items: []v1.Secret{*secret},
		}

		return true, result, nil
	}

	fakeClient.AddReactor("list", "secrets", listReactor)
	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))
	store := newSecretCache(fakeClient, fakeClock, time.Minute)

	key := objectKey{namespace: "ns", name: "name"}
	itemExists := func() (bool, error) {
		store.lock.Lock()
		defer store.lock.Unlock()
		_, ok := store.items[key]
		return ok, nil
	}

	reflectorRunning := func() bool {
		store.lock.Lock()
		defer store.lock.Unlock()
		item := store.items[key]

		item.lock.Lock()
		defer item.lock.Unlock()
		return !item.stopped
	}

	reflectorInitialized := func() (bool, error) {
		store.lock.Lock()
		defer store.lock.Unlock()
		item := store.items[key]

		item.lock.Lock()
		defer item.lock.Unlock()
		return item.store.hasSynced(), nil
	}

	// AddReference should start reflector.
	store.AddReference("ns", "name", "pod")
	if err := wait.Poll(10*time.Millisecond, 10*time.Second, itemExists); err != nil {
		t.Errorf("item wasn't added to cache")
	}

	fakeClock.Step(90 * time.Second)
	store.startRecycleIdleWatch()

	// Reflector didn't yet initialize, so it shouldn't be stopped.
	// However, Get should still be failing.
	assert.True(t, reflectorRunning())
	initialized, _ := reflectorInitialized()
	assert.False(t, initialized)
	_, err := store.Get("ns", "name")
	if err == nil || !strings.Contains(err.Error(), "failed to sync") {
		t.Errorf("Expected failed to sync error, got: %v", err)
	}

	// Initialization should successfully finish.
	fakeClock.Step(30 * time.Second)
	if err := wait.Poll(10*time.Millisecond, time.Second, reflectorInitialized); err != nil {
		t.Errorf("reflector didn't iniailize correctly")
	}

	// recycling shouldn't stop the reflector because it was accessed within last minute.
	store.startRecycleIdleWatch()
	assert.True(t, reflectorRunning())

	obj, _ := store.Get("ns", "name")
	assert.True(t, apiequality.Semantic.DeepEqual(secret, obj))
}

func TestRefMapHandlesReferencesCorrectly(t *testing.T) {
	secret1 := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret1",
			Namespace: "ns1",
		},
	}
	type step struct {
		action         string
		ns             string
		name           string
		referencedFrom types.UID
	}
	type expect struct {
		ns             string
		name           string
		referencedFrom types.UID
		expectCount    int
	}
	tests := []struct {
		desc    string
		steps   []step
		expects []expect
	}{
		{
			desc: "adding and deleting should works",
			steps: []step{
				{"add", "ns1", "secret1", "pod1"},
				{"add", "ns1", "secret1", "pod1"},
				{"delete", "ns1", "secret1", "pod1"},
				{"delete", "ns1", "secret1", "pod1"},
			},
			expects: []expect{
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 2},
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 0},
			},
		},
		{
			desc: "deleting a non-existent reference should have no effect",
			steps: []step{
				{"delete", "ns1", "secret1", "pod1"},
			},
			expects: []expect{
				{"ns1", "secret1", "pod1", 0},
			},
		},
		{
			desc: "deleting more than adding should not lead to negative refcount",
			steps: []step{
				{"add", "ns1", "secret1", "pod1"},
				{"delete", "ns1", "secret1", "pod1"},
				{"delete", "ns1", "secret1", "pod1"},
			},
			expects: []expect{
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 0},
				{"ns1", "secret1", "pod1", 0},
			},
		},
		{
			desc: "deleting should not affect refcount of other objects or referencedFrom",
			steps: []step{
				{"add", "ns1", "secret1", "pod1"},
				{"delete", "ns1", "secret1", "pod2"},
				{"delete", "ns1", "secret2", "pod1"},
				{"delete", "ns2", "secret1", "pod1"},
			},
			expects: []expect{
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 1},
				{"ns1", "secret1", "pod1", 1},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			fakeClient := &fake.Clientset{}
			listReactor := func(a core.Action) (bool, runtime.Object, error) {
				result := &v1.SecretList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "200",
					},
					Items: []v1.Secret{*secret1},
				}
				return true, result, nil
			}
			fakeClient.AddReactor("list", "secrets", listReactor)
			fakeWatch := watch.NewFake()
			fakeClient.AddWatchReactor("secrets", core.DefaultWatchReactor(fakeWatch, nil))
			fakeClock := testingclock.NewFakeClock(time.Now())
			store := newSecretCache(fakeClient, fakeClock, time.Minute)

			for i, step := range tc.steps {
				expect := tc.expects[i]
				switch step.action {
				case "add":
					store.AddReference(step.ns, step.name, step.referencedFrom)
				case "delete":
					store.DeleteReference(step.ns, step.name, step.referencedFrom)
				default:
					t.Errorf("unrecognized action of testcase %v", tc.desc)
				}

				key := objectKey{namespace: expect.ns, name: expect.name}
				item, exists := store.items[key]
				if !exists {
					if tc.expects[i].expectCount != 0 {
						t.Errorf("reference to %v/%v from %v should exists", expect.ns, expect.name, expect.referencedFrom)
					}
				} else if item.refMap[expect.referencedFrom] != expect.expectCount {
					t.Errorf("expects %v but got %v", expect.expectCount, item.refMap[expect.referencedFrom])
				}
			}
		})
	}
}
