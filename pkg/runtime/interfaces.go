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

package runtime

// Decoder defines methods for deserializing API objects into a given type
type Decoder interface {
	Decode(data []byte) (Object, error)
	DecodeInto(data []byte, obj Object) error
}

// Encoder defines methods for serializing API objects into bytes
type Encoder interface {
	Encode(obj Object) (data []byte, err error)
}

// Codec defines methods for serializing and deserializing API objects.
type Codec interface {
	Decoder
	Encoder
}

// ResourceVersioner provides methods for setting and retrieving
// the resource version from an API object.
type ResourceVersioner interface {
	SetResourceVersion(obj Object, version uint64) error
	ResourceVersion(obj Object) (uint64, error)
}

// SelfLinker provides methods for setting and retrieving the SelfLink field of an API object.
type SelfLinker interface {
	SetSelfLink(obj Object, selfLink string) error
	SelfLink(obj Object) (string, error)

	// Knowing ID is sometimes necssary to use a SelfLinker.
	ID(obj Object) (string, error)
}

// All api types must support the Object interface. It's deliberately tiny so that this is not an onerous
// burden. Implement it with a pointer reciever; this will allow us to use the go compiler to check the
// one thing about our objects that it's capable of checking for us.
type Object interface {
	// This function is used only to enforce membership. It's never called.
	// TODO: Consider mass rename in the future to make it do something useful.
	IsAnAPIObject()
}
