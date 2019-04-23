/*
Copyright 2019 The Kubernetes Authors.

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

package cacher

import (
	"bytes"
	"io"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// FIXME:
type serializationResult struct {
	once sync.Once

	// raw is serializaed object.
	raw []byte
	// err is error from serialization.
	err error
}

// FIXME:
type CachingVersionedObject struct {
	// FIXME:
	Object runtime.Object

	// FIXME: ???
	lock sync.Mutex
	// FIXME: sync.Map looks like a good usecase for us:
	// - StoreAndLoad a function that computes stuff
	// - the function that computes has sync.Once underneath.
	serialization map[runtime.Encoder]*serializationResult
}

// FIXME:
func (o *CachingVersionedObject) Encode(e runtime.WithVersionEncoder, w io.Writer) error {
	serialization := func() *serializationResult {
		// FIXME: It probably would benefit to have a fast path,
		//   instead of always locking.

		o.lock.Lock()
		defer o.lock.Unlock()

		serialization, exists := o.serialization[e.Encoder]
		if exists {
			return serialization
		}

		serialization = &serializationResult{}
		o.serialization[e.Encoder] = serialization
		return serialization
	}()
	serialization.once.Do(func() {
		buffer := bytes.NewBuffer(nil)
		serialization.err = e.Encoder.Encode(o.Object.DeepCopyObject(), buffer)
		serialization.raw = buffer.Bytes()
	})
	// Once invoked, fields of serialization will not change.
	if serialization.err != nil {
		return serialization.err
	}
	_, err := w.Write(serialization.raw)
	return err
}

func (o *CachingVersionedObject) GetObjectKind() schema.ObjectKind {
	return o.Object.GetObjectKind()
}

func (o *CachingVersionedObject) DeepCopyObject() runtime.Object {
	// FIXME:
	panic("CachingVersionedObject should never be deep-copied")
	return o
}

// FIXME: Add comment.
type CachingObject struct {
	// Object is the object (potentially in the internal version)
	// for which serializations are cached when computed.
	Object runtime.Object

	// FIXME: ???
	lock sync.Mutex
	// FIXME: maybe sync.Map also here?
	// Maybe it's overkill, but we probably need some fast-path...
	Versioned map[runtime.GroupVersioner]*CachingVersionedObject
}


// FIXME:
// - ensure that appropriate parameters are passed
// - object is unversioned?
// - maybe couple more invariants?
func (o *CachingObject) Encode(e runtime.WithVersionEncoder, w io.Writer) error {
	gvks, isUnversioned, err := e.ObjectTyper.ObjectKinds(o.Object)
	if err != nil {
		return err
	}
	encodeVersion := e.Version
	if isUnversioned {
		encodeVersion = nil
	}

	versioned, err := func() (*CachingVersionedObject, error) {
		// FIXME: It probably would benefit to have a fast path,
		//   instead of always locking.

		o.lock.Lock()
		defer o.lock.Unlock()

		versioned, exists := o.Versioned[e.Version]
		if exists {
			return versioned, nil
		}

		versioned = &CachingVersionedObject{}
		if encodeVersion == nil {
			versioned.Object = o.Object.DeepCopyObject()
			versioned.Object.GetObjectKind().SetGroupVersionKind(gvks[0])
		} else {
			versioned.Object, err = e.ObjectConvertor.ConvertToVersion(o.Object.DeepCopyObject(), encodeVersion)
			if err != nil {
				return nil, err
			}
			// Conversion is responsible for setting the proper group, version, and kind onto the outgoing object
		}
		if ne, ok := versioned.Object.(runtime.NestedObjectEncoder); ok {
			if err := ne.EncodeNestedObjects(runtime.WithVersionEncoder{Version: encodeVersion, Encoder: e.Encoder, ObjectTyper: e.ObjectTyper}); err != nil {
				return nil, err
			}
		}
		o.Versioned[encodeVersion] = versioned
		return versioned, nil
	}()
	if err != nil {
		return err
	}
	return e.Encoder.Encode(versioned, w)
}

func (o *CachingObject) GetObjectKind() schema.ObjectKind {
	return o.Object.GetObjectKind()
}

func (o *CachingObject) DeepCopyObject() runtime.Object {
	// FIXME:
	panic("CachingObject should never be deep-copied")
	return o
}
