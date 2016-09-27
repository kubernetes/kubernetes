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

package serializer

import (
	"k8s.io/client-go/1.5/pkg/runtime"
)

// TODO: We should figure out what happens when someone asks
// encoder for version and it conflicts with the raw serializer.
type negotiatedSerializerWrapper struct {
	info       runtime.SerializerInfo
	streamInfo runtime.StreamSerializerInfo
}

func NegotiatedSerializerWrapper(info runtime.SerializerInfo, streamInfo runtime.StreamSerializerInfo) runtime.NegotiatedSerializer {
	return &negotiatedSerializerWrapper{info, streamInfo}
}

func (n *negotiatedSerializerWrapper) SupportedMediaTypes() []string {
	return []string{}
}

func (n *negotiatedSerializerWrapper) SerializerForMediaType(mediaType string, options map[string]string) (runtime.SerializerInfo, bool) {
	return n.info, true
}

func (n *negotiatedSerializerWrapper) SupportedStreamingMediaTypes() []string {
	return []string{}
}

func (n *negotiatedSerializerWrapper) StreamingSerializerForMediaType(mediaType string, options map[string]string) (runtime.StreamSerializerInfo, bool) {
	return n.streamInfo, true
}

func (n *negotiatedSerializerWrapper) EncoderForVersion(e runtime.Encoder, _ runtime.GroupVersioner) runtime.Encoder {
	return e
}

func (n *negotiatedSerializerWrapper) DecoderToVersion(d runtime.Decoder, _gv runtime.GroupVersioner) runtime.Decoder {
	return d
}
