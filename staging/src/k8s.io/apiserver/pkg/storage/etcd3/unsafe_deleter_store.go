/*
Copyright 2024 The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/value"

	"k8s.io/klog/v2"
)

// NewStoreWithUnsafeCorruptObjectDeletion, returns a store with
// AllowUnsafeCorruptObjectDeletion enabled, for deletion, it will make an
// attempt to perform the normal deletion flow, but if either of the below
// occurs:
// a) the data (associated with the resource being deleted) retrieved
// from the storage failed to transform properly (eg. decryption failure)
// b) the data (associated with the resource being deleted) failed to
// decode properly (eg. corrupt data)
// it will disregard these errors, bypass the finalzer constraints,
// deletion hook(s) and go ahead with the deletion flow.
//
// WARNING: This will break the cluster if the resource has dependencies
// Use only if you REALLY know what you are doing.
// WARNING: Vendor(s) will most likely consider using this option to be
// in violation of the support of their product.
// The default value is false, and the user must opt in to enable it.
func NewStoreWithUnsafeCorruptObjectDeletion(delegate storage.Interface) storage.Interface {
	return &corruptObjectDeleter{Interface: delegate}
}

// NewErrorInterpretingDecoder decorates the given decoder, it determines if the
// error returned by the given decoder is of interest (eg. the object is
// undecodable) and then it wraps the returned error appropriately so
// the store can decide whether the object is a candidate for unsafe delete.
func NewErrorInterpretingDecoder(decoder storage.Decoder) storage.Decoder {
	return &errorInterpretingDecoder{Decoder: decoder}
}

// NewErrorInterpretingTransformer decorates the given transformer, it determines
// if the error returned by the given transformer is of interest (eg. the data from
// the storage is untransformable) and then it wraps the returned error appropriately
// so the store can decide whether the object is a candidate for unsafe delete.
func NewErrorInterpretingTransformer(transformer value.Transformer) value.Transformer {
	return &errorInterpretingTransformer{Transformer: transformer}
}

// corruptObjectDeleter abstracts unsafe deletion flow for etcd3 storage
type corruptObjectDeleter struct {
	storage.Interface
}

func (s *corruptObjectDeleter) Delete(
	ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions,
	validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	// let's try to go through the normal deletion flow first
	ignoreStoreReadErr := opts.IgnoreStoreReadError
	opts.IgnoreStoreReadError = false
	err := s.Interface.Delete(ctx, key, out, preconditions, validateDeletion, cachedExistingObject, opts)
	if err == nil {
		return nil
	}

	// normal deletion flow failed, should we ignore store read error?
	if !ignoreStoreReadErr {
		return err
	}
	var corruptObjErr *corruptObjectError
	if !errors.As(err, &corruptObjErr) {
		klog.ErrorS(err, "the error does not represent a corrupt object, not proceeding with unsafe delete", "key", key)
		return err
	}

	// if we are here:
	// - the normal deletion flow failed
	// - the user has set IgnoreStoreReadError in the delete options, and
	// - and the err represents a corrupt object due to
	//   either transformation or decode failure
	// ignore the error and try the unsafe delete operation
	klog.ErrorS(corruptObjErr, "normal deletion flow failed, trying with unsafe delete", "key", key)
	opts.IgnoreStoreReadError = true
	return s.Interface.Delete(ctx, key, out, preconditions, validateDeletion, cachedExistingObject, opts)
}

func (s *corruptObjectDeleter) Get(ctx context.Context, key string, opts storage.GetOptions, out runtime.Object) error {
	if err := s.Interface.Get(ctx, key, opts, out); err != nil {
		var corruptObjErr *corruptObjectError
		if !errors.As(err, &corruptObjErr) {
			// this error does not represent a corrupt object
			return err
		}
		return corruptObjErr.NewStorageError(key)
	}
	return nil
}

func (s *corruptObjectDeleter) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	aggregateErr := &aggregatedStorageError{resourcePrefix: "list"}
	opts.AggregateCorruptObjFn = func(itemKey string, err error) bool {
		// TODO: make it configurable?
		if len(aggregateErr.errs) > 100 {
			return true
		}
		var corruptObjErr *corruptObjectError
		if !errors.As(err, &corruptObjErr) {
			// this error does not represent a corrupt object,
			// so we will abort the list operation
			return true
		}

		aggregateErr.errs = append(aggregateErr.errs, corruptObjErr.NewStorageError(itemKey))
		return false
	}

	err := s.Interface.GetList(ctx, key, opts, listObj)

	if len(aggregateErr.errs) > 0 {
		// we have aggregated a list of corrupt objects
		klog.V(5).ErrorS(aggregateErr, "listing corrupt objectss")
		return aggregateErr
	}
	return err
}

// errorInterpretingDecoder wraps the error returned by the decorated decoder
type errorInterpretingDecoder struct {
	storage.Decoder
}

func (d *errorInterpretingDecoder) Decode(codec runtime.Codec, versioner storage.Versioner, value []byte, objPtr runtime.Object, rev int64) error {
	// TODO: right now any error is deemed as undecodable, in the future, we
	// can apply some filter, if need be.
	if err := d.Decoder.Decode(codec, versioner, value, objPtr, rev); err != nil {
		return &corruptObjectError{err: err, errType: Undecodable}
	}
	return nil
}

// decodeListItem decodes bytes value in array into object.
func (d *errorInterpretingDecoder) DecodeListItem(ctx context.Context, data []byte, rev uint64, codec runtime.Codec, versioner storage.Versioner, newItemFunc func() runtime.Object) (runtime.Object, error) {
	// TODO: right now any error is deemed as undecodable, in the future, we
	// can apply some filter, if need be.
	obj, err := d.Decoder.DecodeListItem(ctx, data, rev, codec, versioner, newItemFunc)
	if err != nil {
		err = &corruptObjectError{err: err, errType: Undecodable}
	}
	return obj, err
}

// errorInterpretingTransformer wraps the error returned by the transformer
type errorInterpretingTransformer struct {
	value.Transformer
}

func (t *errorInterpretingTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	// TODO: right now any error is deemed as undecodable, in the future, we
	// can apply some filter, if need be. For example, any network error
	out, stale, err := t.Transformer.TransformFromStorage(ctx, data, dataCtx)
	if err != nil {
		err = &corruptObjectError{err: err, errType: Untransformable}
	}
	return out, stale, err
}

// corruptObjectError is used as a marker, the store will apply the unsafe
// deletion flow only if the error returned is a corruptObjectError type
type corruptObjectError struct {
	err     error
	errType int
}

const (
	Untransformable int = iota + 1
	Undecodable
)

var typeToMessage = map[int]string{
	Untransformable: "data from the storage is not transformable",
	Undecodable:     "object not decodable",
}

func (e *corruptObjectError) Unwrap() error { return e.err }
func (e *corruptObjectError) Error() string {
	return fmt.Sprintf("%s: %v", typeToMessage[e.errType], e.err)
}

func (e *corruptObjectError) NewStorageError(key string) *storage.StorageError {
	return storage.NewCorruptObjError(key, typeToMessage[e.errType], e.err)
}

// aggregatedStorageError holds an aggregated list of storage.StorageError
type aggregatedStorageError struct {
	resourcePrefix string
	errs           []*storage.StorageError
}

func (e *aggregatedStorageError) Error() string {
	if len(e.errs) == 0 {
		return ""
	}
	if len(e.errs) == 1 {
		return e.errs[0].Error()
	}

	var b strings.Builder
	fmt.Fprintf(&b, "unable to transform or decode %d objects: {\n", len(e.errs))
	for _, err := range e.errs {
		fmt.Fprintf(&b, "\t%s\n", err.Error())
	}
	b.WriteString("}")
	return b.String()
}
