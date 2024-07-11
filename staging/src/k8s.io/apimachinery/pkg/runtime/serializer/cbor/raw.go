/*
Copyright 2024 The Kubernetes Authors.

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

package cbor

import (
	"fmt"
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
)

var sharedTranscoders transcoders

var rawTypeTranscodeFuncs = map[reflect.Type]func(reflect.Value) error{
	reflect.TypeFor[runtime.RawExtension](): func(rv reflect.Value) error {
		if !rv.CanAddr() {
			return nil
		}
		re := rv.Addr().Interface().(*runtime.RawExtension)
		if re.Raw == nil {
			// When Raw is nil it encodes to null. Don't change nil Raw values during
			// transcoding, they would have unmarshalled from JSON as nil too.
			return nil
		}
		j, err := re.MarshalJSON()
		if err != nil {
			return fmt.Errorf("failed to transcode RawExtension to JSON: %w", err)
		}
		re.Raw = j
		return nil
	},
}

func transcodeRawTypes(v interface{}) error {
	if v == nil {
		return nil
	}

	rv := reflect.ValueOf(v)
	return sharedTranscoders.getTranscoder(rv.Type()).fn(rv)
}

type transcoder struct {
	fn func(rv reflect.Value) error
}

var noop = transcoder{
	fn: func(reflect.Value) error {
		return nil
	},
}

type transcoders struct {
	lock sync.RWMutex
	m    map[reflect.Type]**transcoder
}

func (ts *transcoders) getTranscoder(rt reflect.Type) transcoder {
	ts.lock.RLock()
	tpp, ok := ts.m[rt]
	ts.lock.RUnlock()
	if ok {
		return **tpp
	}

	ts.lock.Lock()
	defer ts.lock.Unlock()
	tp := ts.getTranscoderLocked(rt)
	return *tp
}

func (ts *transcoders) getTranscoderLocked(rt reflect.Type) *transcoder {
	if tpp, ok := ts.m[rt]; ok {
		// A transcoder for this type was cached while waiting to acquire the lock.
		return *tpp
	}

	// Cache the transcoder now, before populating fn, so that circular references between types
	// don't overflow the call stack.
	t := new(transcoder)
	if ts.m == nil {
		ts.m = make(map[reflect.Type]**transcoder)
	}
	ts.m[rt] = &t

	for rawType, fn := range rawTypeTranscodeFuncs {
		if rt == rawType {
			t = &transcoder{fn: fn}
			return t
		}
	}

	switch rt.Kind() {
	case reflect.Array:
		te := ts.getTranscoderLocked(rt.Elem())
		rtlen := rt.Len()
		if rtlen == 0 || te == &noop {
			t = &noop
			break
		}
		t.fn = func(rv reflect.Value) error {
			for i := 0; i < rtlen; i++ {
				if err := te.fn(rv.Index(i)); err != nil {
					return err
				}
			}
			return nil
		}
	case reflect.Interface:
		// Any interface value might have a dynamic type involving RawExtension. It needs to
		// be checked.
		t.fn = func(rv reflect.Value) error {
			if rv.IsNil() {
				return nil
			}
			rv = rv.Elem()
			// The interface element's type is dynamic so its transcoder can't be
			// determined statically.
			return ts.getTranscoder(rv.Type()).fn(rv)
		}
	case reflect.Map:
		rtk := rt.Key()
		tk := ts.getTranscoderLocked(rtk)
		rte := rt.Elem()
		te := ts.getTranscoderLocked(rte)
		if tk == &noop && te == &noop {
			t = &noop
			break
		}
		t.fn = func(rv reflect.Value) error {
			iter := rv.MapRange()
			rvk := reflect.New(rtk).Elem()
			rve := reflect.New(rte).Elem()
			for iter.Next() {
				rvk.SetIterKey(iter)
				if err := tk.fn(rvk); err != nil {
					return err
				}
				rve.SetIterValue(iter)
				if err := te.fn(rve); err != nil {
					return err
				}
			}
			return nil
		}
	case reflect.Pointer:
		te := ts.getTranscoderLocked(rt.Elem())
		if te == &noop {
			t = &noop
			break
		}
		t.fn = func(rv reflect.Value) error {
			if rv.IsNil() {
				return nil
			}
			return te.fn(rv.Elem())
		}
	case reflect.Slice:
		te := ts.getTranscoderLocked(rt.Elem())
		if te == &noop {
			t = &noop
			break
		}
		t.fn = func(rv reflect.Value) error {
			for i := 0; i < rv.Len(); i++ {
				if err := te.fn(rv.Index(i)); err != nil {
					return err
				}
			}
			return nil
		}
	case reflect.Struct:
		type fieldTranscoder struct {
			Index      int
			Transcoder *transcoder
		}
		var fieldTranscoders []fieldTranscoder
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			tf := ts.getTranscoderLocked(f.Type)
			if tf == &noop {
				continue
			}
			fieldTranscoders = append(fieldTranscoders, fieldTranscoder{Index: i, Transcoder: tf})
		}
		if len(fieldTranscoders) == 0 {
			t = &noop
			break
		}
		t.fn = func(rv reflect.Value) error {
			for _, ft := range fieldTranscoders {
				if err := ft.Transcoder.fn(rv.Field(ft.Index)); err != nil {
					return err
				}
			}
			return nil
		}
	default:
		t = &noop
	}

	return t
}
