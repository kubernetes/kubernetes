/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// Interface defines how clients may interact with the objects of an API.
// This is the contract a standard API package must expose as package
// methods.
// TODO: move me to a package that needs to consume me.
type Interface interface {
	// New returns a new API object of the given version ("" for internal
	// representation) and name, or an error if it hasn't been registered.
	NewObject(versionName, typeName string) (interface{}, error)
	// Convert will attempt to convert in into out. Both must be pointers to API objects.
	// For easy testing of conversion functions. Returns an error if the conversion isn't
	// possible.
	Convert(in, out interface{}) error
	// Encode turns the given api object into an appropriate JSON string.
	// Will return an error if the object doesn't have an embedded JSONBase.
	// Obj may be a pointer to a struct, or a struct. If a struct, a copy
	// must be made. If a pointer, the object may be modified before encoding,
	// but will be put back into its original state before returning.
	//
	// Memory/wire format differences:
	//  * Having to keep track of the Kind and APIVersion fields makes tests
	//    very annoying, so the rule is that they are set only in wire format
	//    (json), not when in native (memory) format. This is possible because
	//    both pieces of information are implicit in the go typed object.
	//     * An exception: note that, if there are embedded API objects of known
	//       type, for example, PodList{... Items []Pod ...}, these embedded
	//       objects must be of the same version of the object they are embedded
	//       within, and their APIVersion and Kind must both be empty.
	//     * Note that the exception does not apply to the APIObject type, which
	//       recursively does Encode()/Decode(), and is capable of expressing any
	//       API object.
	//  * Only versioned objects should be encoded. This means that, if you pass
	//    a native object, Encode will convert it to a versioned object. For
	//    example, an api.Pod will get converted to a v1beta1.Pod. However, if
	//    you pass in an object that's already versioned (v1beta1.Pod), Encode
	//    will not modify it.
	//
	// The purpose of the above complex conversion behavior is to allow us to
	// change the memory format yet not break compatibility with any stored
	// objects, whether they be in our storage layer (e.g., etcd), or in user's
	// config files.
	//
	// TODO/next steps: When we add our second versioned type, this package will
	// need a version of Encode that lets you choose the wire version. A configurable
	// default will be needed, to allow operating in clusters that haven't yet
	// upgraded.
	//
	Encode(obj interface{}) (data []byte, err error)
	// Decode converts a YAML or JSON string back into a pointer to an api object.
	// Deduces the type based upon the APIVersion and Kind fields, which are set
	// by Encode. Only versioned objects (APIVersion != "") are accepted. The object
	// will be converted into the in-memory unversioned type before being returned.
	Decode(data []byte) (interface{}, error)
	// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
	// if data.Kind is set and doesn't match the type of obj. Obj should be a
	// pointer to an api type.
	// If obj's APIVersion doesn't match that in data, an attempt will be made to convert
	// data into obj's version.
	DecodeInto(data []byte, obj interface{}) error
}

type RegisterHandler func(*conversion.Scheme)

func New(version string, types ...RegisterHandler) Interface {
	scheme := conversion.NewScheme()
	scheme.InternalVersion = ""
	scheme.ExternalVersion = version
	for _, h := range types {
		h(scheme)
	}
	return scheme
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(api Interface, obj interface{}) string {
	bytes, err := api.Encode(obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}
