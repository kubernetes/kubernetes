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

package storage

import (
	"fmt"
	"testing"

	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/api"
)

type fakeStorage struct {
	Interface
	calledWithSuggestion    bool
	calledWithoutSuggestion bool
}

func (fs *fakeStorage) GuaranteedUpdate(
	ctx context.Context,
	key string,
	ptrToType runtime.Object,
	ignoreNotFound bool,
	preconditions *Preconditions,
	tryUpdate UpdateFunc,
	suggestion ...runtime.Object,
) error {
	if len(suggestion) == 1 {
		fs.calledWithSuggestion = true
		return errors.NewConflict(api.SchemeGroupVersion.WithResource("pod").GroupResource(), "name", fmt.Errorf("foo"))
	}
	fs.calledWithoutSuggestion = true
	return nil
}

func TestGuaranteedUpdateDropsSuggestionOnConflict(t *testing.T) {
	keyFunc := func(obj runtime.Object) (string, error) {
		return NamespaceKeyFunc("/pods", obj)
	}
	getAttrsFunc := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod := obj.(*api.Pod)
		return labels.Set{"name": pod.Name}, nil, nil
	}
	watchCache := newWatchCache(10, keyFunc, getAttrsFunc)
	storage := &fakeStorage{}
	cacher := &Cacher{
		copier:     api.Scheme,
		watchCache: watchCache,
		storage:    storage,
	}

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "foo",
		},
	}
	if err := watchCache.Add(pod); err != nil {
		t.Fatalf("unable to add to cache: %v", err)
	}

	err := cacher.GuaranteedUpdate(context.Background(), "/pods/default/foo", &api.Pod{}, false, nil, nil)
	if err != nil {
		t.Errorf("couldn't update: %v", err)
	}
	if !storage.calledWithSuggestion {
		t.Errorf("c.storage.GuaranteedUpdate not called with a suggestion")
	}
	if !storage.calledWithoutSuggestion {
		t.Errorf("c.storage.GuaranteedUpdate not called without a suggestion after a conflict")
	}
}
