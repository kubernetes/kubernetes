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

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// NegotiateError is returned when a ClientNegotiator is unable to locate
// a serializer for the requested operation.
type NegotiateError struct {
	ContentType string
	Stream      bool
}

func (e NegotiateError) Error() string {
	if e.Stream {
		return fmt.Sprintf("no stream serializers registered for %s", e.ContentType)
	}
	return fmt.Sprintf("no serializers registered for %s", e.ContentType)
}

type clientNegotiator struct {
	serializer     NegotiatedSerializer
	encode, decode GroupVersioner
}

func (n *clientNegotiator) Encoder(contentType string, params map[string]string) (Encoder, error) {
	// TODO: `pretty=1` is handled in NegotiateOutputMediaType, consider moving it to this method
	// if client negotiators truly need to use it
	mediaTypes := n.serializer.SupportedMediaTypes()
	info, ok := SerializerInfoForMediaType(mediaTypes, contentType)
	if !ok {
		if len(contentType) != 0 || len(mediaTypes) == 0 {
			return nil, NegotiateError{ContentType: contentType}
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
			return nil, NegotiateError{ContentType: contentType}
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
			return nil, nil, nil, NegotiateError{ContentType: contentType, Stream: true}
		}
		info = mediaTypes[0]
	}
	if info.StreamSerializer == nil {
		return nil, nil, nil, NegotiateError{ContentType: info.MediaType, Stream: true}
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
