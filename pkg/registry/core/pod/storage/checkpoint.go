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

package storage

import (
	"context"
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func newCheckpointStorage(store rest.StandardStorage) *CheckpointREST {
	return &CheckpointREST{store: store}
}

// CheckpointREST implements the REST endpoint for checkpointing pods on nodes
type CheckpointREST struct {
	store rest.StandardStorage
}

var _ = rest.NamedCreater(&CheckpointREST{})
var _ = rest.GroupVersionKindProvider(&CheckpointREST{})

// GroupVersionKind specifies a particular GroupVersionKind to discovery
func (r *CheckpointREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "policy", Version: "v1beta1", Kind: "Checkpoint"}
}

// New creates a new checkpoint resource
func (r *CheckpointREST) New() runtime.Object {
	return &policy.Checkpoint{}
}

// Create attempts to create a new checkpoint.
func (r *CheckpointREST) Create(ctx context.Context, name string, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	checkpoint, ok := obj.(*policy.Checkpoint)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("not a Checkpoint object: %T", obj))
	}

	if name != checkpoint.Name {
		return nil, errors.NewBadRequest("name in URL does not match name in Checkpoint object")
	}

	// Success!
	return &metav1.Status{Status: metav1.StatusSuccess}, nil
}

func (r *CheckpointREST) Get(ctx context.Context, name string, opts runtime.Object) (runtime.Object, error) {
	return nil, nil
}

func (r *CheckpointREST) Connect(ctx context.Context, name string, opts runtime.Object, responder rest.Responder) (http.Handler, error) {
	return nil, nil
}
