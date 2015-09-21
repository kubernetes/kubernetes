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

package runtime

// Note that the types provided in this file are not versioned and are intended to be
// safe to use from within all versions of every API object.

// TypeMeta is shared by all top level objects. The proper way to use it is to inline it in your type,
// like this:
// type MyAwesomeAPIObject struct {
//      runtime.TypeMeta    `json:",inline"`
//      ... // other fields
// }
// func (*MyAwesomeAPIObject) IsAnAPIObject() {}
//
// TypeMeta is provided here for convenience. You may use it directly from this package or define
// your own with the same fields.
//
type TypeMeta struct {
	APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
}

// PluginBase is like TypeMeta, but it's intended for plugin objects that won't ever be encoded
// except while embedded in other objects.
type PluginBase struct {
	Kind string `json:"kind,omitempty"`
}

// EmbeddedObject has appropriate encoder and decoder functions, such that on the wire, it's
// stored as a []byte, but in memory, the contained object is accessible as an Object
// via the Get() function. Only valid API objects may be stored via EmbeddedObject.
// The purpose of this is to allow an API object of type known only at runtime to be
// embedded within other API objects.
//
// Note that object assumes that you've registered all of your api types with the api package.
//
// EmbeddedObject and RawExtension can be used together to allow for API object extensions:
// see the comment for RawExtension.
type EmbeddedObject struct {
	Object
}

// RawExtension is used with EmbeddedObject to do a two-phase encoding of extension objects.
//
// To use this, make a field which has RawExtension as its type in your external, versioned
// struct, and EmbeddedObject in your internal struct. You also need to register your
// various plugin types.
//
// // Internal package:
// type MyAPIObject struct {
// 	runtime.TypeMeta `json:",inline"`
//	MyPlugin runtime.EmbeddedObject `json:"myPlugin"`
// }
// type PluginA struct {
// 	runtime.PluginBase `json:",inline"`
//	AOption string `json:"aOption"`
// }
//
// // External package:
// type MyAPIObject struct {
// 	runtime.TypeMeta `json:",inline"`
//	MyPlugin runtime.RawExtension `json:"myPlugin"`
// }
// type PluginA struct {
// 	runtime.PluginBase `json:",inline"`
//	AOption string `json:"aOption"`
// }
//
// // On the wire, the JSON will look something like this:
// {
//	"kind":"MyAPIObject",
//	"apiVersion":"v1beta1",
//	"myPlugin": {
//		"kind":"PluginA",
//		"aOption":"foo",
//	},
// }
//
// So what happens? Decode first uses json or yaml to unmarshal the serialized data into
// your external MyAPIObject. That causes the raw JSON to be stored, but not unpacked.
// The next step is to copy (using pkg/conversion) into the internal struct. The runtime
// package's DefaultScheme has conversion functions installed which will unpack the
// JSON stored in RawExtension, turning it into the correct object type, and storing it
// in the EmbeddedObject. (TODO: In the case where the object is of an unknown type, a
// runtime.Unknown object will be created and stored.)
type RawExtension struct {
	RawJSON []byte
}

// Unknown allows api objects with unknown types to be passed-through. This can be used
// to deal with the API objects from a plug-in. Unknown objects still have functioning
// TypeMeta features-- kind, version, etc.
// TODO: Make this object have easy access to field based accessors and settors for
// metadata and field mutatation.
type Unknown struct {
	TypeMeta `json:",inline"`
	// RawJSON will hold the complete JSON of the object which couldn't be matched
	// with a registered type. Most likely, nothing should be done with this
	// except for passing it through the system.
	RawJSON []byte
}

func (*Unknown) IsAnAPIObject() {}

// Unstructured allows objects that do not have Golang structs registered to be manipulated
// generically. This can be used to deal with the API objects from a plug-in. Unstructured
// objects still have functioning TypeMeta features-- kind, version, etc.
// TODO: Make this object have easy access to field based accessors and settors for
// metadata and field mutatation.
type Unstructured struct {
	TypeMeta `json:",inline"`
	// Object is a JSON compatible map with string, float, int, []interface{}, or map[string]interface{}
	// children.
	Object map[string]interface{}
}

func (*Unstructured) IsAnAPIObject() {}
