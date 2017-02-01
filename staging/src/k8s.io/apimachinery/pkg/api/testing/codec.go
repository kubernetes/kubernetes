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

package testing

import (
	"os"
	"mime"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var testCodecMediaType string

// TestCodec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API_TYPE env var.
func TestCodec(codecs runtimeserializer.CodecFactory, gvs ...schema.GroupVersion) runtime.Codec {
	if len(testCodecMediaType) != 0 {
		serializer, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), testCodecMediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", testCodecMediaType))
		}
		return codecs.CodecForVersions(serializer.Serializer, codecs.UniversalDeserializer(), schema.GroupVersions(gvs), nil)
	}
	return codecs.LegacyCodec(gvs...)
}

func init() {
	if apiMediaType := os.Getenv("KUBE_TEST_API_TYPE"); len(apiMediaType) > 0 {
		var err error
		testCodecMediaType, _, err = mime.ParseMediaType(apiMediaType)
		if err != nil {
			panic(err)
		}
	}
}
