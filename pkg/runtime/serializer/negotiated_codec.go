/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package serializer

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// TODO: We should figure out what happens when someone asks
// encoder for version and it conflicts with the raw serializer.
type negotiatedSerializerWrapper struct {
	serializer          runtime.Serializer
	streamingSerializer runtime.Serializer
	framer              runtime.Framer
}

func NegotiatedSerializerWrapper(serializer, streamingSerializer runtime.Serializer, framer runtime.Framer) runtime.NegotiatedSerializer {
	return &negotiatedSerializerWrapper{serializer, streamingSerializer, framer}
}

func (n *negotiatedSerializerWrapper) SupportedMediaTypes() []string {
	return []string{}
}

func (n *negotiatedSerializerWrapper) SerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, bool) {
	return n.serializer, true
}

func (n *negotiatedSerializerWrapper) SupportedStreamingMediaTypes() []string {
	return []string{}
}

func (n *negotiatedSerializerWrapper) StreamingSerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, runtime.Framer, string, bool) {
	return n.streamingSerializer, n.framer, "", true
}

func (n *negotiatedSerializerWrapper) EncoderForVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Encoder {
	return n.serializer
}

func (n *negotiatedSerializerWrapper) DecoderToVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Decoder {
	return n.serializer
}
