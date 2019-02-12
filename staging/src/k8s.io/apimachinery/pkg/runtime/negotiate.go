/*
Copyright 2019 The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type clientNegotiator struct {
	serializer     NegotiatedSerializer
	encode, decode GroupVersioner
}

func (n *clientNegotiator) Encoder(contentType string, params map[string]string) (Encoder, error) {
	mediaTypes := n.serializer.SupportedMediaTypes()
	info, ok := SerializerInfoForMediaType(mediaTypes, contentType)
	if !ok {
		if len(contentType) != 0 || len(mediaTypes) == 0 {
			return nil, fmt.Errorf("no serializers registered for %s", contentType)
		}
		info = mediaTypes[0]
	}
	return n.serializer.EncoderForVersion(info.Serializer, n.encode), nil
}

func (n *clientNegotiator) Decoder(contentType string, params map[string]string) (Decoder, error) {
	mediaTypes := n.serializer.SupportedMediaTypes()
	info, ok := SerializerInfoForMediaType(mediaTypes, contentType)
	if !ok {
		if len(contentType) != 0 || len(mediaTypes) == 0 {
			return nil, fmt.Errorf("no serializers registered for %s", contentType)
		}
		info = mediaTypes[0]
	}
	return n.serializer.DecoderToVersion(info.Serializer, n.decode), nil
}

func (n *clientNegotiator) StreamDecoder(contentType string, params map[string]string) (Decoder, Serializer, Framer, error) {
	mediaTypes := n.serializer.SupportedMediaTypes()
	info, ok := SerializerInfoForMediaType(mediaTypes, contentType)
	if !ok {
		if len(contentType) != 0 || len(mediaTypes) == 0 {
			return nil, nil, nil, fmt.Errorf("no stream serializers registered for %s", contentType)
		}
		info = mediaTypes[0]
	}
	if info.StreamSerializer == nil {
		return nil, nil, nil, fmt.Errorf("no stream serializers registered for %s", info.MediaType)
	}
	return n.serializer.DecoderToVersion(info.Serializer, n.decode), info.StreamSerializer.Serializer, info.StreamSerializer.Framer, nil
}

// NewClientNegotiator will attempt to retrieve the appropriate encoder, decoder, or
// stream decoder for a given content type. Does not perform any conversion, but will
// encode the object to the desired group, version, and kind. Use when creating a client.
func NewClientNegotiator(serializer NegotiatedSerializer, gv schema.GroupVersion) ClientNegotiator {
	return &clientNegotiator{
		serializer: serializer,
		encode:     gv,
	}
}

// NewInternalClientNegotiator applies the default client rules for connecting to a Kubernetes apiserver
// where objects are converted to gv prior to sending and decoded to their internal representation prior
// to retrieval.
//
// DEPRECATED: Internal clients are deprecated and will be removed in a future Kubernetes release.
func NewInternalClientNegotiator(serializer NegotiatedSerializer, gv schema.GroupVersion) ClientNegotiator {
	decode := schema.GroupVersions{
		{
			Group:   gv.Group,
			Version: APIVersionInternal,
		},
		// always include the legacy group as a decoding target to handle non-error `Status` return types
		{
			Group:   "",
			Version: APIVersionInternal,
		},
	}
	return &clientNegotiator{
		encode:     gv,
		decode:     decode,
		serializer: serializer,
	}
}

// NewSimpleClientNegotiator will negotiate for a single serializer. This should only be used
// for testing or when the caller is taking responsibility for setting the GVK on encoded objects.
func NewSimpleClientNegotiator(info SerializerInfo, gv schema.GroupVersion) ClientNegotiator {
	return &clientNegotiator{
		serializer: &simpleNegotiatedSerializer{info: info},
		encode:     gv,
	}
}

type simpleNegotiatedSerializer struct {
	info SerializerInfo
}

func NewSimpleNegotiatedSerializer(info SerializerInfo) NegotiatedSerializer {
	return &simpleNegotiatedSerializer{info: info}
}

func (n *simpleNegotiatedSerializer) SupportedMediaTypes() []SerializerInfo {
	return []SerializerInfo{n.info}
}

func (n *simpleNegotiatedSerializer) EncoderForVersion(e Encoder, _ GroupVersioner) Encoder {
	return e
}

func (n *simpleNegotiatedSerializer) DecoderToVersion(d Decoder, _gv GroupVersioner) Decoder {
	return d
}

// DirectEncoder serializes an object and ensures the GVK is set.
type DirectEncoder struct {
	Version GroupVersioner
	Encoder
	ObjectTyper
}

// Encode does not do conversion. It sets the gvk during serialization.
func (e DirectEncoder) Encode(obj Object, stream io.Writer) error {
	gvks, _, err := e.ObjectTyper.ObjectKinds(obj)
	if err != nil {
		if IsNotRegisteredError(err) {
			return e.Encoder.Encode(obj, stream)
		}
		return err
	}
	kind := obj.GetObjectKind()
	oldGVK := kind.GroupVersionKind()
	gvk := gvks[0]
	if e.Version != nil {
		preferredGVK, ok := e.Version.KindForGroupVersionKinds(gvks)
		if ok {
			gvk = preferredGVK
		}
	}
	kind.SetGroupVersionKind(gvk)
	err = e.Encoder.Encode(obj, stream)
	kind.SetGroupVersionKind(oldGVK)
	return err
}

// DirectDecoder clears the group version kind of a deserialized object.
type DirectDecoder struct {
	Decoder
}

// Decode does not do conversion. It removes the gvk during deserialization.
func (d DirectDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into Object) (Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := d.Decoder.Decode(data, defaults, into)
	if obj != nil {
		kind := obj.GetObjectKind()
		// clearing the gvk is just a convention of a codec
		kind.SetGroupVersionKind(schema.GroupVersionKind{})
	}
	return obj, gvk, err
}
