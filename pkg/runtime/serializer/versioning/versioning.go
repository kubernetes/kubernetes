/*
Copyright 2014 The Kubernetes Authors.

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

package versioning

import (
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// NewCodecForScheme is a convenience method for callers that are using a scheme.
func NewCodecForScheme(
	// TODO: I should be a scheme interface?
	scheme *runtime.Scheme,
	encoder runtime.Encoder,
	decoder runtime.Decoder,
	encodeVersion runtime.GroupVersioner,
	decodeVersion runtime.GroupVersioner,
) runtime.Codec {
	return NewCodec(encoder, decoder, runtime.UnsafeObjectConvertor(scheme), scheme, scheme, scheme, encodeVersion, decodeVersion)
}

// NewCodec takes objects in their internal versions and converts them to external versions before
// serializing them. It assumes the serializer provided to it only deals with external versions.
// This class is also a serializer, but is generally used with a specific version.
func NewCodec(
	encoder runtime.Encoder,
	decoder runtime.Decoder,
	convertor runtime.ObjectConvertor,
	creater runtime.ObjectCreater,
	copier runtime.ObjectCopier,
	typer runtime.ObjectTyper,
	encodeVersion runtime.GroupVersioner,
	decodeVersion runtime.GroupVersioner,
) runtime.Codec {
	internal := &codec{
		encoder:   encoder,
		decoder:   decoder,
		convertor: convertor,
		creater:   creater,
		copier:    copier,
		typer:     typer,

		encodeVersion: encodeVersion,
		decodeVersion: decodeVersion,
	}
	return internal
}

type codec struct {
	encoder   runtime.Encoder
	decoder   runtime.Decoder
	convertor runtime.ObjectConvertor
	creater   runtime.ObjectCreater
	copier    runtime.ObjectCopier
	typer     runtime.ObjectTyper

	encodeVersion runtime.GroupVersioner
	decodeVersion runtime.GroupVersioner
}

// Decode attempts a decode of the object, then tries to convert it to the internal version. If into is provided and the decoding is
// successful, the returned runtime.Object will be the value passed as into. Note that this may bypass conversion if you pass an
// into that matches the serialized version.
func (c *codec) Decode(data []byte, defaultGVK *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	versioned, isVersioned := into.(*runtime.VersionedObjects)
	if isVersioned {
		into = versioned.Last()
	}

	obj, gvk, err := c.decoder.Decode(data, defaultGVK, into)
	if err != nil {
		return nil, gvk, err
	}

	// if we specify a target, use generic conversion.
	if into != nil {
		if into == obj {
			if isVersioned {
				return versioned, gvk, nil
			}
			return into, gvk, nil
		}
		if err := c.convertor.Convert(obj, into); err != nil {
			return nil, gvk, err
		}
		if isVersioned {
			versioned.Objects = []runtime.Object{obj, into}
			return versioned, gvk, nil
		}
		return into, gvk, nil
	}

	// Convert if needed.
	if isVersioned {
		// create a copy, because ConvertToVersion does not guarantee non-mutation of objects
		copied, err := c.copier.Copy(obj)
		if err != nil {
			copied = obj
		}
		versioned.Objects = []runtime.Object{copied}
	}
	out, err := c.convertor.ConvertToVersion(obj, c.decodeVersion)
	if err != nil {
		return nil, gvk, err
	}
	if isVersioned {
		if versioned.Last() != out {
			versioned.Objects = append(versioned.Objects, out)
		}
		return versioned, gvk, nil
	}
	return out, gvk, nil
}

// Encode ensures the provided object is output in the appropriate group and version, invoking
// conversion if necessary. Unversioned objects (according to the ObjectTyper) are output as is.
func (c *codec) Encode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case *runtime.Unknown:
		if gv, ok := runtime.PreferredGroupVersion(c.encodeVersion); ok {
			t.APIVersion = gv.String()
		}
		return c.encoder.Encode(obj, w)
	case *runtime.Unstructured:
		if gv, ok := runtime.PreferredGroupVersion(c.encodeVersion); ok {
			t.SetAPIVersion(gv.String())
		}
		return c.encoder.Encode(obj, w)
	case *runtime.UnstructuredList:
		if gv, ok := runtime.PreferredGroupVersion(c.encodeVersion); ok {
			t.SetAPIVersion(gv.String())
		}
		return c.encoder.Encode(obj, w)
	}

	gvks, isUnversioned, err := c.typer.ObjectKinds(obj)
	if err != nil {
		return err
	}

	if c.encodeVersion == nil || isUnversioned {
		objectKind := obj.GetObjectKind()
		old := objectKind.GroupVersionKind()
		objectKind.SetGroupVersionKind(gvks[0])
		err = c.encoder.Encode(obj, w)
		objectKind.SetGroupVersionKind(old)
		return err
	}

	// Perform a conversion if necessary
	objectKind := obj.GetObjectKind()
	old := objectKind.GroupVersionKind()
	out, err := c.convertor.ConvertToVersion(obj, c.encodeVersion)
	if err != nil {
		return err
	}
	// Conversion is responsible for setting the proper group, version, and kind onto the outgoing object
	err = c.encoder.Encode(out, w)
	// restore the old GVK, in case conversion returned the same object
	objectKind.SetGroupVersionKind(old)
	return err
}
