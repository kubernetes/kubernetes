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

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"

	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"

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

func newSecretCache(fakeClient clientset.Interface) *objectCache {
	return &objectCache{
		listObject:    listSecret(fakeClient),
		watchObject:   watchSecret(fakeClient),
		newObject:     func() runtime.Object { return &v1.Secret{} },
		isImmutable:   isSecretImmutable,
		groupResource: corev1.Resource("secret"),
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

	store := newSecretCache(fakeClient)

	store.AddReference("ns", "name")
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

	store.DeleteReference("ns", "name")
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

	store := newSecretCache(fakeClient)

	store.AddReference("ns", "name")
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
		store.AddReference("ns", "name")
		store.DeleteReference("ns", "name")
	}
	actions := fakeClient.Actions()
	assert.Equal(t, 2, len(actions), "unexpected actions: %#v", actions)

	// Final delete also doesn't trigger any action.
	store.DeleteReference("ns", "name")
	_, err := store.Get("ns", "name")
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Errorf("unexpected error: %v", err)
	}
	actions = fakeClient.Actions()
	assert.Equal(t, 2, len(actions), "unexpected actions: %#v", actions)
}

func TestImmutableSecretStopsTheReflector(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ImmutableEphemeralVolumes, true)()

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

			store := newSecretCache(fakeClient)

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
				select {
				case <-item.stopCh:
					return false
				default:
					return true
				}
			}

			// AddReference should start reflector.
			store.AddReference("ns", "name")
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
