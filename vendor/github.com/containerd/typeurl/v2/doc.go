/*
   Copyright The containerd Authors.

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

package typeurl

// Package typeurl assists with managing the registration, marshaling, and
// unmarshaling of types encoded as protobuf.Any.
//
// A protobuf.Any is a proto message that can contain any arbitrary data. It
// consists of two components, a TypeUrl and a Value, and its proto definition
// looks like this:
//
//   message Any {
//     string type_url = 1;
//     bytes value = 2;
//   }
//
// The TypeUrl is used to distinguish the contents from other proto.Any
// messages. This typeurl library manages these URLs to enable automagic
// marshaling and unmarshaling of the contents.
//
// For example, consider this go struct:
//
//   type Foo struct {
//     Field1 string
//     Field2 string
//   }
//
// To use typeurl, types must first be registered. This is typically done in
// the init function
//
//   func init() {
//      typeurl.Register(&Foo{}, "Foo")
//   }
//
// This will register the type Foo with the url path "Foo". The arguments to
// Register are variadic, and are used to construct a url path. Consider this
// example, from the github.com/containerd/containerd/client package:
//
//   func init() {
//     const prefix = "types.containerd.io"
//     // register TypeUrls for commonly marshaled external types
//     major := strconv.Itoa(specs.VersionMajor)
//     typeurl.Register(&specs.Spec{}, prefix, "opencontainers/runtime-spec", major, "Spec")
//     // this function has more Register calls, which are elided.
//   }
//
// This registers several types under a more complex url, which ends up mapping
// to `types.containerd.io/opencontainers/runtime-spec/1/Spec` (or some other
// value for major).
//
// Once a type is registered, it can be marshaled to a proto.Any message simply
// by calling `MarshalAny`, like this:
//
//   foo := &Foo{Field1: "value1", Field2: "value2"}
//   anyFoo, err := typeurl.MarshalAny(foo)
//
// MarshalAny will resolve the correct URL for the type. If the type in
// question implements the proto.Message interface, then it will be marshaled
// as a proto message. Otherwise, it will be marshaled as json. This means that
// typeurl will work on any arbitrary data, whether or not it has a proto
// definition, as long as it can be serialized to json.
//
// To unmarshal, the process is simply inverse:
//
//   iface, err := typeurl.UnmarshalAny(anyFoo)
//   foo := iface.(*Foo)
//
// The correct type is automatically chosen from the type registry, and the
// returned interface can be cast straight to that type.
