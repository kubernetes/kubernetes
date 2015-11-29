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

// NewSerializedConversions handles the conversion between Object and RawExtension given
// a supplied serializer. If conversion is desired, the caller provide a serializer
// that performs opinionated conversions.
func NewSerializedConversions(serializer Serializer) *conversion.ConversionFuncs {
	fns := conversion.NewConversionFuncs()
	fns.Register(
		func(in Object, out *RawExtension, s conversion.Scope) error {
			if in == nil {
				out.RawJSON = []byte("null")
				return nil
			}
			data, err := Encode(serializer, in)
			if err != nil {
				return err
			}
			out.RawJSON = data
			return nil
		},

		func(in *RawExtension, out *Object, s conversion.Scope) error {
			if len(in.RawJSON) == 0 || (len(in.RawJSON) == 4 && string(in.RawJSON) == "null") {
				*out = nil
				return nil
			}
			// Figure out the type and kind of the output object.
			outVersion := s.Meta().DestVersion
			gv, err := unversioned.ParseGroupVersion(outVersion)
			if err != nil {
				return err
			}
			obj, _, err := serializer.Decode(in.RawJSON, &unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version})
			if err != nil {
				return err
			}
			*out = obj
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
				// Figure out the type and kind of the output object.
				outVersion := s.Meta().DestVersion
				gv, err := unversioned.ParseGroupVersion(outVersion)
				if err != nil {
					return err
				}
				data := src[i].RawJSON
				obj, _, err := serializer.Decode(data, &unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version})
				if err == nil {
					dest[i] = obj
				} else {
					dest[i] = &Unknown{
						RawJSON: data,
					}
				}
			}
			*out = dest
			return nil
		},
	)
	return &fns
}
