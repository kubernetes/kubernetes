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

package scheme

import (
	"os"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// All kubectl code should eventually switch to use this Registry and Scheme instead of the global ones.

// GroupFactoryRegistry is the APIGroupFactoryRegistry (overlaps a bit with Registry, see comments in package for details)
var GroupFactoryRegistry = make(announced.APIGroupFactoryRegistry)

// Registry is an instance of an API registry.
var Registry = registered.NewOrDie(os.Getenv("KUBE_API_VERSIONS"))

// Scheme is the default instance of runtime.Scheme to which types in the Kubernetes API are already registered.
var Scheme = runtime.NewScheme()

// Codecs provides access to encoding and decoding for the scheme
var Codecs = serializer.NewCodecFactory(Scheme)

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(Scheme)

// Versions is a list of group versions in order of preferred serialization.  This used to be discovered dynamically,
// from the server for use in the client, but that gives conflicting lists of non-existent versions.  This only needs to
// live until we stop attempting to perform any conversion client-side and is only valid for items existent in our scheme.
var Versions = []schema.GroupVersion{}

// DefaultJSONEncoder returns a default encoder for our scheme
func DefaultJSONEncoder() runtime.Encoder {
	return Codecs.LegacyCodec(Versions...)
}
