// +build proto

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/protobuf"
)

// contentTypeProtobuf is the protobuf type exposed for Kubernetes. It is private to prevent others from
// depending on it unintentionally.
// TODO: potentially move to pkg/api (since it's part of the Kube public API) and pass it in to the
//   CodecFactory on initialization.
const contentTypeProtobuf = "application/vnd.kubernetes.protobuf"

func protobufSerializer(scheme *runtime.Scheme) (serializerType, bool) {
	serializer := protobuf.NewSerializer(scheme, runtime.ObjectTyperToTyper(scheme), contentTypeProtobuf)
	raw := protobuf.NewRawSerializer(scheme, runtime.ObjectTyperToTyper(scheme), contentTypeProtobuf)
	return serializerType{
		AcceptContentTypes: []string{contentTypeProtobuf},
		ContentType:        contentTypeProtobuf,
		FileExtensions:     []string{"pb"},
		Serializer:         serializer,
		RawSerializer:      raw,
	}, true
}

func init() {
	serializerExtensions = append(serializerExtensions, protobufSerializer)
}
