/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import (
	"context"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func newTokenCleaner() (*TokenCleaner, *fake.Clientset, coreinformers.SecretInformer, error) {
	options := DefaultTokenCleanerOptions()
	cl := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(cl, options.SecretResync)
	secrets := informerFactory.Core().V1().Secrets()
	tcc, err := NewTokenCleaner(cl, secrets, options)
	if err != nil {
		return nil, nil, nil, err
	}
	return tcc, cl, secrets, nil
}

func TestCleanerNoExpiration(t *testing.T) {
	cleaner, cl, secrets, err := newTokenCleaner()
	if err != nil {
		t.Fatalf("error creating TokenCleaner: %v", err)
	}

	secret := newTokenSecret("tokenID", "tokenSecret")
	secrets.Informer().GetIndexer().Add(secret)

	cleaner.evalSecret(context.TODO(), secret)

	expected := []core.Action{}

	verifyActions(t, expected, cl.Actions())
}

func TestCleanerExpired(t *testing.T) {
	cleaner, cl, secrets, err := newTokenCleaner()
	if err != nil {
		t.Fatalf("error creating TokenCleaner: %v", err)
	}

	secret := newTokenSecret("tokenID", "tokenSecret")
	addSecretExpiration(secret, timeString(-time.Hour))
	secrets.Informer().GetIndexer().Add(secret)

	cleaner.evalSecret(context.TODO(), secret)

	expected := []core.Action{
		core.NewDeleteActionWithOptions(
			schema.GroupVersionResource{Version: "v1", Resource: "secrets"},
			api.NamespaceSystem,
			secret.ObjectMeta.Name,
			metav1.DeleteOptions{
				Preconditions: metav1.NewUIDPreconditions(string(secret.UID)),
			}),
	}

	verifyActions(t, expected, cl.Actions())
}

func TestCleanerNotExpired(t *testing.T) {
	cleaner, cl, secrets, err := newTokenCleaner()
	if err != nil {
		t.Fatalf("error creating TokenCleaner: %v", err)
	}

	secret := newTokenSecret("tokenID", "tokenSecret")
	addSecretExpiration(secret, timeString(time.Hour))
	secrets.Informer().GetIndexer().Add(secret)

	cleaner.evalSecret(context.TODO(), secret)

	expected := []core.Action{}

	verifyActions(t, expected, cl.Actions())
}

func TestCleanerExpiredAt(t *testing.T) {
	cleaner, cl, secrets, err := newTokenCleaner()
	if err != nil {
		t.Fatalf("error creating TokenCleaner: %v", err)
	}

	secret := newTokenSecret("tokenID", "tokenSecret")
	addSecretExpiration(secret, timeString(2*time.Second))
	secrets.Informer().GetIndexer().Add(secret)
	cleaner.enqueueSecrets(secret)
	expected := []core.Action{}
	verifyFunc := func() {
		cleaner.processNextWorkItem(context.TODO())
		verifyActions(t, expected, cl.Actions())
	}
	// token has not expired currently
	verifyFunc()

	if cleaner.queue.Len() != 0 {
		t.Errorf("not using the queue, the length should be 0, now: %v", cleaner.queue.Len())
	}

	var conditionFunc = func() (bool, error) {
		if cleaner.queue.Len() == 1 {
			return true, nil
		}
		return false, nil
	}

	err = wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, conditionFunc)
	if err != nil {
		t.Fatalf("secret is put back into the queue, the queue length should be 1, error: %v\n", err)
	}

	// secret was eventually deleted
	expected = []core.Action{
		core.NewDeleteActionWithOptions(
			schema.GroupVersionResource{Version: "v1", Resource: "secrets"},
			api.NamespaceSystem,
			secret.ObjectMeta.Name,
			metav1.DeleteOptions{
				Preconditions: metav1.NewUIDPreconditions(string(secret.UID)),
			}),
	}
	verifyFunc()
}
