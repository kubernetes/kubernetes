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
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	endpointsrequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"

	"k8s.io/klog/v2"
)

// NewDefaultDecoder returns the default decoder for etcd3 store
func NewDefaultDecoder(codec runtime.Codec, versioner storage.Versioner) Decoder {
	return &defaultDecoder{
		codec:     codec,
		versioner: versioner,
	}
}

// Decoder is used by the etcd storage implementation to decode
// transformed data from the storage into an object
type Decoder interface {
	// Decode decodes value of bytes into object. It will also
	// set the object resource version to rev.
	// On success, objPtr would be set to the object.
	Decode(value []byte, objPtr runtime.Object, rev int64) error

	// DecodeListItem decodes bytes value in array into object.
	DecodeListItem(ctx context.Context, data []byte, rev uint64, newItemFunc func() runtime.Object) (runtime.Object, error)
}

var _ Decoder = &defaultDecoder{}

type defaultDecoder struct {
	codec     runtime.Codec
	versioner storage.Versioner
}

// decode decodes value of bytes into object. It will also set the object resource version to rev.
// On success, objPtr would be set to the object.
func (d *defaultDecoder) Decode(value []byte, objPtr runtime.Object, rev int64) error {
	if _, err := conversion.EnforcePtr(objPtr); err != nil {
		// nolint:errorlint // this code was moved from store.go as is
		return fmt.Errorf("unable to convert output object to pointer: %v", err)
	}
	_, _, err := d.codec.Decode(value, nil, objPtr)
	if err != nil {
		return err
	}
	// being unable to set the version does not prevent the object from being extracted
	if err := d.versioner.UpdateObject(objPtr, uint64(rev)); err != nil {
		klog.Errorf("failed to update object version: %v", err)
	}
	return nil
}

// decodeListItem decodes bytes value in array into object.
func (d *defaultDecoder) DecodeListItem(ctx context.Context, data []byte, rev uint64, newItemFunc func() runtime.Object) (runtime.Object, error) {
	startedAt := time.Now()
	defer func() {
		endpointsrequest.TrackDecodeLatency(ctx, time.Since(startedAt))
	}()

	obj, _, err := d.codec.Decode(data, nil, newItemFunc())
	if err != nil {
		return nil, err
	}

	if err := d.versioner.UpdateObject(obj, rev); err != nil {
		klog.Errorf("failed to update object version: %v", err)
	}

	return obj, nil
}
