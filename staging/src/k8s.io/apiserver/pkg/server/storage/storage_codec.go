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
	"mime"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// StorageCodecConfig are the arguments passed to newStorageCodecFn
type StorageCodecConfig struct {
	StorageMediaType  string
	StorageSerializer runtime.StorageSerializer
	StorageVersion    schema.GroupVersion
	MemoryVersion     schema.GroupVersion
	Config            storagebackend.Config

	EncoderDecoratorFn func(runtime.Encoder) runtime.Encoder
	DecoderDecoratorFn func([]runtime.Decoder) []runtime.Decoder
}

// NewStorageCodec assembles a storage codec for the provided storage media type, the provided serializer, and the requested
// storage and memory versions.
func NewStorageCodec(opts StorageCodecConfig) (runtime.Codec, error) {
	mediaType, _, err := mime.ParseMediaType(opts.StorageMediaType)
	if err != nil {
		return nil, fmt.Errorf("%q is not a valid mime-type", opts.StorageMediaType)
	}

	serializer, ok := runtime.SerializerInfoForMediaType(opts.StorageSerializer.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unable to find serializer for %q", mediaType)
	}

	s := serializer.Serializer

	// Give callers the opportunity to wrap encoders and decoders.  For decoders, each returned decoder will
	// be passed to the recognizer so that multiple decoders are available.
	var encoder runtime.Encoder = s
	if opts.EncoderDecoratorFn != nil {
		encoder = opts.EncoderDecoratorFn(encoder)
	}
	decoders := []runtime.Decoder{
		// selected decoder as the primary
		s,
		// universal deserializer as a fallback
		opts.StorageSerializer.UniversalDeserializer(),
		// base64-wrapped universal deserializer as a last resort.
		// this allows reading base64-encoded protobuf, which should only exist if etcd2+protobuf was used at some point.
		// data written that way could exist in etcd2, or could have been migrated to etcd3.
		// TODO: flag this type of data if we encounter it, require migration (read to decode, write to persist using a supported encoder), and remove in 1.8
		runtime.NewBase64Serializer(nil, opts.StorageSerializer.UniversalDeserializer()),
	}
	if opts.DecoderDecoratorFn != nil {
		decoders = opts.DecoderDecoratorFn(decoders)
	}

	// Ensure the storage receives the correct version.
	encoder = opts.StorageSerializer.EncoderForVersion(
		encoder,
		runtime.NewMultiGroupVersioner(
			opts.StorageVersion,
			schema.GroupKind{Group: opts.StorageVersion.Group},
			schema.GroupKind{Group: opts.MemoryVersion.Group},
		),
	)
	decoder := opts.StorageSerializer.DecoderToVersion(
		recognizer.NewDecoder(decoders...),
		runtime.NewCoercingMultiGroupVersioner(
			opts.MemoryVersion,
			schema.GroupKind{Group: opts.MemoryVersion.Group},
			schema.GroupKind{Group: opts.StorageVersion.Group},
		),
	)

	return runtime.NewCodec(encoder, decoder), nil
}
