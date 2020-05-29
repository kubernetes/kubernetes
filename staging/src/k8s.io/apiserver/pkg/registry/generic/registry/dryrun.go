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

func (s *DryRunnableStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, deleteValidation storage.ValidateObjectFunc, dryRun bool) error {
	if dryRun {
		if err := s.Storage.Get(ctx, key, storage.GetOptions{}, out); err != nil {
			return err
		}
		if err := preconditions.Check(key, out); err != nil {
			return err
		}
		return deleteValidation(ctx, out)
	}
	return s.Storage.Delete(ctx, key, out, preconditions, deleteValidation)
}

func (s *DryRunnableStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	return s.Storage.Watch(ctx, key, opts)
}

func (s *DryRunnableStorage) WatchList(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	return s.Storage.WatchList(ctx, key, opts)
}

func (s *DryRunnableStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	return s.Storage.Get(ctx, key, opts, objPtr)
}

func (s *DryRunnableStorage) GetToList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	return s.Storage.GetToList(ctx, key, opts, listObj)
}

func (s *DryRunnableStorage) List(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	return s.Storage.List(ctx, key, opts, listObj)
}

func (s *DryRunnableStorage) GuaranteedUpdate(
	ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, dryRun bool, suggestion ...runtime.Object) error {
	if dryRun {
		err := s.Storage.Get(ctx, key, storage.GetOptions{IgnoreNotFound: ignoreNotFound}, ptrToType)
		if err != nil {
			return err
		}
		err = preconditions.Check(key, ptrToType)
		if err != nil {
			return err
		}
		rev, err := s.Versioner().ObjectResourceVersion(ptrToType)
		if err != nil {
			return err
		}
		out, _, err := tryUpdate(ptrToType, storage.ResponseMeta{ResourceVersion: rev})
		if err != nil {
			return err
		}
		return s.copyInto(out, ptrToType)
	}
	return s.Storage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate, suggestion...)
}

func (s *DryRunnableStorage) Count(key string) (int64, error) {
	return s.Storage.Count(key)
}

func (s *DryRunnableStorage) copyInto(in, out runtime.Object) error {
	var data []byte

	data, err := runtime.Encode(s.Codec, in)
	if err != nil {
		return err
	}
	_, _, err = s.Codec.Decode(data, nil, out)
	if err != nil {
		return err
	}
	return nil

}
