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

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
)

func defaultGVForScope(s conversion.Scope) *unversioned.GroupVersionKind {
	dest := s.Meta().SrcVersion
	if gv, err := unversioned.ParseGroupVersion(dest); err == nil {
		return &unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version}
	}
	return nil
}

// NewSerializedConversions handles the conversion between Object and RawExtension given
// a supplied serializer. If conversion is desired, the caller provide a serializer
// that performs opinionated conversions.
func NewSerializedConversions(serializer Serializer) *conversion.ConversionFuncs {
	fns := conversion.NewConversionFuncs()
	err := fns.Register(
		func(in *Object, out *RawExtension, s conversion.Scope) error {
			if in == nil {
				out.RawJSON = []byte("null")
				return nil
			}
			obj := *in
			if unk, ok := obj.(*Unknown); ok {
				if out.RawJSON != nil {
					out.RawJSON = unk.RawJSON
					return nil
				}
				obj = out.Object
			}
			if obj == nil {
				out.RawJSON = []byte("null")
				return nil
			}
			data, err := Encode(serializer, obj)
			if err != nil {
				return err
			}
			out.RawJSON = data
			return nil
		},

		func(in *RawExtension, out *Object, s conversion.Scope) error {
			data := in.RawJSON
			if len(data) == 0 || (len(data) == 4 && string(data) == "null") {
				*out = nil
				return nil
			}
			obj, gvk, err := serializer.Decode(data, defaultGVForScope(s), nil)
			if err == nil {
				*out = obj
			} else {
				unk := &Unknown{
					RawJSON: data,
				}
				if gvk != nil {
					unk.TypeMeta.Kind = gvk.Kind
					unk.TypeMeta.APIVersion = gvk.GroupVersion().String()
				}
				*out = unk
			}
			return nil
		},

		// Takes a list of objects and encodes them as RawExtension in the output version
		// defined by the conversion.Scope. If objects must be encoded to different schema versions than the default, you
		// should encode them yourself with runtime.Unknown, or convert the object prior to invoking conversion. Objects
		// outside of the current scheme must be added as runtime.Unknown.
		func(in *[]Object, out *[]RawExtension, s conversion.Scope) error {
			src := *in
			dest := make([]RawExtension, len(src))

			for i := range src {
				switch t := src[i].(type) {
				case *Unknown:
					// TODO: rename to Raw and copy content type
					dest[i].RawJSON = t.RawJSON
				default:
					data, err := Encode(serializer, t)
					if err != nil {
						return err
					}
					dest[i].RawJSON = data
				}
			}
			*out = dest
			return nil
		},

		// Attempts to decode objects from the array - if they are unrecognized objects,
		// they are added as Unknown.
		func(in *[]RawExtension, out *[]Object, s conversion.Scope) error {
			src := *in
			dest := make([]Object, len(src))

			for i := range src {
				data := src[i].RawJSON
				obj, gvk, err := serializer.Decode(data, defaultGVForScope(s), nil)
				if err == nil {
					dest[i] = obj
				} else {
					unk := &Unknown{
						RawJSON: data,
					}
					if gvk != nil {
						unk.TypeMeta.Kind = gvk.Kind
						unk.TypeMeta.APIVersion = gvk.GroupVersion().String()
					}
					dest[i] = unk
				}
			}
			*out = dest
			return nil
		},
	)
	if err != nil {
		panic(err)
	}
	return &fns
}
