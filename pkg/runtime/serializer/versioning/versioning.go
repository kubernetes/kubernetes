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
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// EnableCrossGroupDecoding modifies the given decoder in place, if it is a codec
// from this package. It allows objects from one group to be auto-decoded into
// another group. 'destGroup' must already exist in the codec.
// TODO: this is an encapsulation violation and should be refactored
func EnableCrossGroupDecoding(d runtime.Decoder, sourceGroup, destGroup string) error {
	internal, ok := d.(*codec)
	if !ok {
		return fmt.Errorf("unsupported decoder type")
	}

	dest, ok := internal.decodeVersion[destGroup]
	if !ok {
		return fmt.Errorf("group %q is not a possible destination group in the given codec", destGroup)
	}
	internal.decodeVersion[sourceGroup] = dest

	return nil
}

// EnableCrossGroupEncoding modifies the given encoder in place, if it is a codec
// from this package. It allows objects from one group to be auto-decoded into
// another group. 'destGroup' must already exist in the codec.
// TODO: this is an encapsulation violation and should be refactored
func EnableCrossGroupEncoding(e runtime.Encoder, sourceGroup, destGroup string) error {
	internal, ok := e.(*codec)
	if !ok {
		return fmt.Errorf("unsupported encoder type")
	}

	dest, ok := internal.encodeVersion[destGroup]
	if !ok {
		return fmt.Errorf("group %q is not a possible destination group in the given codec", destGroup)
	}
	internal.encodeVersion[sourceGroup] = dest

	return nil
}

// NewCodecForScheme is a convenience method for callers that are using a scheme.
func NewCodecForScheme(
	// TODO: I should be a scheme interface?
	scheme *runtime.Scheme,
	encoder runtime.Encoder,
	decoder runtime.Decoder,
	encodeVersion []unversioned.GroupVersion,
	decodeVersion []unversioned.GroupVersion,
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
	encodeVersion []unversioned.GroupVersion,
	decodeVersion []unversioned.GroupVersion,
) runtime.Codec {
	internal := &codec{
		encoder:   encoder,
		decoder:   decoder,
		convertor: convertor,
		creater:   creater,
		copier:    copier,
		typer:     typer,
	}
	if encodeVersion != nil {
		internal.encodeVersion = make(map[string]unversioned.GroupVersion)
		for _, v := range encodeVersion {
			// first one for a group wins.  This is consistent with best to worst order throughout the codebase
			if _, ok := internal.encodeVersion[v.Group]; ok {
				continue
			}
			internal.encodeVersion[v.Group] = v
		}
		if len(internal.encodeVersion) == 1 {
			for _, v := range internal.encodeVersion {
				internal.preferredEncodeVersion = []unversioned.GroupVersion{v}
			}
		}
	}
	if decodeVersion != nil {
		internal.decodeVersion = make(map[string]unversioned.GroupVersion)
		for _, v := range decodeVersion {
			// first one for a group wins.  This is consistent with best to worst order throughout the codebase
			if _, ok := internal.decodeVersion[v.Group]; ok {
				continue
			}
			internal.decodeVersion[v.Group] = v
		}
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

	encodeVersion map[string]unversioned.GroupVersion
	decodeVersion map[string]unversioned.GroupVersion

	preferredEncodeVersion []unversioned.GroupVersion
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

	// invoke a version conversion
	group := gvk.Group
	if defaultGVK != nil {
		group = defaultGVK.Group
	}
	var targetGV unversioned.GroupVersion
	if c.decodeVersion == nil {
		// convert to internal by default
		targetGV.Group = group
		targetGV.Version = runtime.APIVersionInternal
	} else {
		gv, ok := c.decodeVersion[group]
		if !ok {
			// unknown objects are left in their original version
			if isVersioned {
				versioned.Objects = []runtime.Object{obj}
				return versioned, gvk, nil
			}
			return obj, gvk, nil
		}
		targetGV = gv
	}

	if gvk.GroupVersion() == targetGV {
		if isVersioned {
			versioned.Objects = []runtime.Object{obj}
			return versioned, gvk, nil
		}
		return obj, gvk, nil
	}

	if isVersioned {
		// create a copy, because ConvertToVersion does not guarantee non-mutation of objects
		copied, err := c.copier.Copy(obj)
		if err != nil {
			copied = obj
		}
		versioned.Objects = []runtime.Object{copied}
	}

	// Convert if needed.
	out, err := c.convertor.ConvertToVersion(obj, targetGV)
	if err != nil {
		return nil, gvk, err
	}
	if isVersioned {
		versioned.Objects = append(versioned.Objects, out)
		return versioned, gvk, nil
	}
	return out, gvk, nil
}

// Encode ensures the provided object is output in the appropriate group and version, invoking
// conversion if necessary. Unversioned objects (according to the ObjectTyper) are output as is.
func (c *codec) Encode(obj runtime.Object, w io.Writer) error {
	if _, ok := obj.(*runtime.Unknown); ok {
		return c.encoder.Encode(obj, w)
	}
	gvks, isUnversioned, err := c.typer.ObjectKinds(obj)
	if err != nil {
		return err
	}
	gvk := gvks[0]

	if c.encodeVersion == nil || isUnversioned {
		objectKind := obj.GetObjectKind()
		old := objectKind.GroupVersionKind()
		objectKind.SetGroupVersionKind(gvk)
		err = c.encoder.Encode(obj, w)
		objectKind.SetGroupVersionKind(old)
		return err
	}

	targetGV, ok := c.encodeVersion[gvk.Group]

	// attempt a conversion to the sole encode version
	if !ok && c.preferredEncodeVersion != nil {
		ok = true
		targetGV = c.preferredEncodeVersion[0]
	}

	// if no fallback is available, error
	if !ok {
		return fmt.Errorf("the codec does not recognize group %q for kind %q and cannot encode it", gvk.Group, gvk.Kind)
	}

	// Perform a conversion if necessary
	objectKind := obj.GetObjectKind()
	old := objectKind.GroupVersionKind()
	out, err := c.convertor.ConvertToVersion(obj, targetGV)
	if err != nil {
		if ok {
			return err
		}
	} else {
		obj = out
	}
	// Conversion is responsible for setting the proper group, version, and kind onto the outgoing object
	err = c.encoder.Encode(obj, w)
	// restore the old GVK, in case conversion returned the same object
	objectKind.SetGroupVersionKind(old)
	return err
}
