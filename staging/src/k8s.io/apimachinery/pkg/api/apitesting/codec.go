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

package apitesting

import (
	"fmt"
	"mime"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
)

var (
	testCodecMediaType        string
	testStorageCodecMediaType string
)

// TestCodec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API_TYPE env var.
func TestCodec(codecs runtimeserializer.CodecFactory, gvs ...schema.GroupVersion) runtime.Codec {
	if len(testCodecMediaType) != 0 {
		serializerInfo, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), testCodecMediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", testCodecMediaType))
		}
		return codecs.CodecForVersions(serializerInfo.Serializer, codecs.UniversalDeserializer(), schema.GroupVersions(gvs), nil)
	}
	return codecs.LegacyCodec(gvs...)
}

// TestStorageCodec returns the codec for the API version to test against used in storage, as set by the
// KUBE_TEST_API_STORAGE_TYPE env var.
func TestStorageCodec(codecs runtimeserializer.CodecFactory, gvs ...schema.GroupVersion) runtime.Codec {
	if len(testStorageCodecMediaType) != 0 {
		serializerInfo, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), testStorageCodecMediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", testStorageCodecMediaType))
		}

		// etcd2 only supports string data - we must wrap any result before returning
		// TODO: remove for etcd3 / make parameterizable
		serializer := serializerInfo.Serializer
		if !serializerInfo.EncodesAsText {
			serializer = runtime.NewBase64Serializer(serializer, serializer)
		}

		decoder := recognizer.NewDecoder(serializer, codecs.UniversalDeserializer())
		return codecs.CodecForVersions(serializer, decoder, schema.GroupVersions(gvs), nil)

	}
	return codecs.LegacyCodec(gvs...)
}

func init() {
	var err error
	if apiMediaType := os.Getenv("KUBE_TEST_API_TYPE"); len(apiMediaType) > 0 {
		testCodecMediaType, _, err = mime.ParseMediaType(apiMediaType)
		if err != nil {
			panic(err)
		}
	}

	if storageMediaType := os.Getenv("KUBE_TEST_API_STORAGE_TYPE"); len(storageMediaType) > 0 {
		testStorageCodecMediaType, _, err = mime.ParseMediaType(storageMediaType)
		if err != nil {
			panic(err)
		}
	}
}

// InstallOrDieFunc mirrors install functions that require success
type InstallOrDieFunc func(scheme *runtime.Scheme)

// SchemeForInstallOrDie builds a simple test scheme and codecfactory pair for easy unit testing from higher level install methods
func SchemeForInstallOrDie(installFns ...InstallOrDieFunc) (*runtime.Scheme, runtimeserializer.CodecFactory) {
	scheme := runtime.NewScheme()
	codecFactory := runtimeserializer.NewCodecFactory(scheme)
	for _, installFn := range installFns {
		installFn(scheme)
	}

	return scheme, codecFactory
}

// InstallFunc mirrors install functions that can return an error
type InstallFunc func(scheme *runtime.Scheme) error

// SchemeForOrDie builds a simple test scheme and codecfactory pair for easy unit testing from the bare registration methods.
func SchemeForOrDie(installFns ...InstallFunc) (*runtime.Scheme, runtimeserializer.CodecFactory) {
	scheme := runtime.NewScheme()
	codecFactory := runtimeserializer.NewCodecFactory(scheme)
	for _, installFn := range installFns {
		if err := installFn(scheme); err != nil {
			panic(err)
		}
	}

	return scheme, codecFactory
}
