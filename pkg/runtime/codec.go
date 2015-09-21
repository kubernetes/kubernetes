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

// CodecFor returns a Codec that invokes Encode with the provided version.
func CodecFor(scheme *Scheme, version string) Codec {
	return &codecWrapper{scheme, version}
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(codec Codec, obj Object) string {
	bytes, err := codec.Encode(obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
}

// codecWrapper implements encoding to an alternative
// default version for a scheme.
type codecWrapper struct {
	*Scheme
	version string
}

// Encode implements Codec
func (c *codecWrapper) Encode(obj Object) ([]byte, error) {
	return c.Scheme.EncodeToVersion(obj, c.version)
}
