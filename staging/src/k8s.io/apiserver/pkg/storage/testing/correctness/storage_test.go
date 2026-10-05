/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

func newTestStorage() *ModelStorage {
	return NewStorage(NewEmptyModel("", func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, storage.APIObjectVersioner{}))
}

func TestStorageCreate(t *testing.T) {
	// The model stores objects, not bytes, so there is no stored form to validate.
	storagetesting.RunTestCreate(t.Context(), t, newTestStorage(), func(context.Context, *testing.T, string) {})
}

func TestStorageGet(t *testing.T) {
	storagetesting.RunTestGet(t.Context(), t, newTestStorage())
}

func TestStorageGetListRecursivePrefix(t *testing.T) {
	storagetesting.RunTestGetListRecursivePrefix(t.Context(), t, newTestStorage())
}

func TestStorageCreateWithKeyExist(t *testing.T) {
	storagetesting.RunTestCreateWithKeyExist(t.Context(), t, newTestStorage())
}

func TestStorageUnconditionalDelete(t *testing.T) {
	storagetesting.RunTestUnconditionalDelete(t.Context(), t, newTestStorage())
}

func TestStorageConditionalDelete(t *testing.T) {
	storagetesting.RunTestConditionalDelete(t.Context(), t, newTestStorage())
}

func TestStorageDeleteWithSuggestion(t *testing.T) {
	storagetesting.RunTestDeleteWithSuggestion(t.Context(), t, newTestStorage())
}

func TestStorageDeleteWithSuggestionAndConflict(t *testing.T) {
	storagetesting.RunTestDeleteWithSuggestionAndConflict(t.Context(), t, newTestStorage())
}

func TestStorageDeleteWithSuggestionOfDeletedObject(t *testing.T) {
	storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(t.Context(), t, newTestStorage())
}

func TestStoragePreconditionalDeleteWithSuggestion(t *testing.T) {
	storagetesting.RunTestPreconditionalDeleteWithSuggestion(t.Context(), t, newTestStorage())
}

func TestStoragePreconditionalDeleteWithOnlySuggestionPass(t *testing.T) {
	storagetesting.RunTestPreconditionalDeleteWithOnlySuggestionPass(t.Context(), t, newTestStorage())
}

func TestStorageGuaranteedUpdateWithSuggestionAndConflict(t *testing.T) {
	storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(t.Context(), t, newTestStorage())
}
