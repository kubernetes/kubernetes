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

package registry

import (
	"context"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

type DryRunnableStorage struct {
	Storage storage.Interface
	Codec   runtime.Codec
}

func (s *DryRunnableStorage) Versioner() storage.Versioner {
	return s.Storage.Versioner()
}

func (s *DryRunnableStorage) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64, dryRun bool) error {
	if dryRun {
		if err := s.Storage.Get(ctx, key, storage.GetOptions{}, out); err == nil {
			return storage.NewKeyExistsError(key, 0)
		}
		return s.copyInto(obj, out)
	}
	return s.Storage.Create(ctx, key, obj, out, ttl)
}

func (s *DryRunnableStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, deleteValidation storage.ValidateObjectFunc, dryRun bool, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	if dryRun {
		if err := s.Storage.Get(ctx, key, storage.GetOptions{}, out); err != nil {
			return err
		}
		if err := preconditions.Check(key, out); err != nil {
			return err
		}
		return deleteValidation(ctx, out)
	}
	return s.Storage.Delete(ctx, key, out, preconditions, deleteValidation, cachedExistingObject, opts)
}

func (s *DryRunnableStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	return s.Storage.Watch(ctx, key, opts)
}

func (s *DryRunnableStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	return s.Storage.Get(ctx, key, opts, objPtr)
}

func (s *DryRunnableStorage) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	return s.Storage.GetList(ctx, key, opts, listObj)
}

func (s *DryRunnableStorage) GuaranteedUpdate(
	ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, dryRun bool, cachedExistingObject runtime.Object) error {
	if dryRun {
		var current runtime.Object
		v, err := conversion.EnforcePtr(destination)
		if err != nil {
			return fmt.Errorf("unable to convert output object to pointer: %v", err)
		}
		if u, ok := v.Addr().Interface().(runtime.Unstructured); ok {
			current = u.NewEmptyInstance()
		} else {
			current = reflect.New(v.Type()).Interface().(runtime.Object)
		}

		err = s.Storage.Get(ctx, key, storage.GetOptions{IgnoreNotFound: ignoreNotFound}, current)
		if err != nil {
			return err
		}
		err = preconditions.Check(key, current)
		if err != nil {
			return err
		}
		rev, err := s.Versioner().ObjectResourceVersion(current)
		if err != nil {
			return err
		}
		updated, _, err := tryUpdate(current, storage.ResponseMeta{ResourceVersion: rev})
		if err != nil {
			return err
		}
		return s.copyInto(updated, destination)
	}
	return s.Storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, cachedExistingObject)
}

func (s *DryRunnableStorage) Stats(ctx context.Context) (storage.Stats, error) {
	return s.Storage.Stats(ctx)
}

func (s *DryRunnableStorage) copyInto(in, out runtime.Object) error {
	var data []byte

	data, err := runtime.Encode(s.Codec, in)
	if err != nil {
		return err
	}
	_, _, err = s.Codec.Decode(data, nil, out)
	return err
}
