/*
Copyright 2018 The Kubernetes Authors.

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
	"bytes"
	"io"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// SerializationScheme defines how the object should be serialized.
type SerializationScheme struct {
	GV        schema.GroupVersion
	MediaType []string
}

// Serialization contains cached result of single serialization.
type serialization struct {
	once sync.Once
	// mediaType to which the object is serialized.
	mediaType *string
	// raw is serialized object.
	raw []byte
	// err is error from serialization.
	err error
}

func (s *serialization) serialize(object Object, serializer Serializer) {
	s.once.Do(func() {
		buffer := bytes.NewBuffer(nil)
		s.err = serializer.Encode(object.DeepCopyObject(), buffer)
		s.raw = buffer.Bytes()
	})
}

// CachingVersionedObject is an object that caches different serialization
// results for a given versioned object.
type CachingVersionedObject struct {
	Once sync.Once
	// GV is group version of Object.
	GV *schema.GroupVersion
	// Object is an object that can be serialized.
	Object Object
	// Serializations is a list of cached serializations of Object.
	Serializations []*serialization
}

// FIXME: Comment.
func (o *CachingVersionedObject) GetObjectKind() schema.ObjectKind {
	return o.Object.GetObjectKind()
}

// FIXME: Comment.
func (o *CachingVersionedObject) DeepCopyObject() Object {
	// FIXME: Don't panic on it.
	panic("CachingVersionedObject should never be deep-copied")
	return o
}

// FIXME: Add comment.
func NewCachingVersionedObjects(schemes []*SerializationScheme) []*CachingVersionedObject {
	result := make([]*CachingVersionedObject, len(schemes))
	for i, scheme := range schemes {
		obj := &CachingVersionedObject{
			Once:           sync.Once{},
			GV:             &scheme.GV,
			Serializations: make([]*serialization, len(scheme.MediaType)),
		}
		for j := range scheme.MediaType {
			obj.Serializations[j] = &serialization{
				once:      sync.Once{},
				mediaType: &scheme.MediaType[j],
			}
		}
		result[i] = obj
	}
	return result
}

func (o *CachingVersionedObject) serialization(mediaType *string) *serialization {
	for _, s := range o.Serializations {
		if *s.mediaType == *mediaType {
			return s
		}
	}
	return nil
}

// FIXME: Add comment.
func (o *CachingVersionedObject) SetObject(f func() (Object, error)) {
	o.Once.Do(func () {
		obj, err := f()
		if err != nil {
			// TODO: How to handle it better?
			// TODO: Log error.
			o.Serializations = nil
			return
		}
		o.Object = obj
	})
}

// FIXME: Add comment.
func (o *CachingVersionedObject) Serialize(mediaType string, serializer Serializer, w io.Writer) error {
	serialization := o.serialization(&mediaType)
	if serialization == nil {
		return serializer.Encode(o.Object, w)
	}
	serialization.serialize(o.Object, serializer)
	if serialization.err != nil {
		return serialization.err
	}
	_, err := w.Write(serialization.raw)
	return err
}

// FIXME: Add comment.
type CachingObject struct {
	// Object is the object (potentially in the internal version)
	// for which serializations should be cached when computed.
	Object Object
	// Versioned is a list of cached versioned serializations.
	Versioned []*CachingVersionedObject
}

// FIXME: Comment.
func (o *CachingObject) GetObjectKind() schema.ObjectKind {
	return o.Object.GetObjectKind()
}

// FIXME: Comment.
func (o *CachingObject) DeepCopyObject() Object {
	// FIXME: Don't panic on it.
	panic("CachingObject should never be deep-copied")
	return o
}

// FIXME: Comment.
func (o *CachingObject) GetCachingVersionedObject(gv GroupVersioner) *CachingVersionedObject {
	for _, v := range o.Versioned {
		if *v.GV == gv {
			return v
		}
	}
	return nil
}
