// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"reflect"
	"sync"
)

type mapKeyValueEncodeFunc struct {
	kf, ef       encodeFunc
	kpool, vpool sync.Pool
}

func (me *mapKeyValueEncodeFunc) encodeKeyValues(e *bytes.Buffer, em *encMode, v reflect.Value, kvs []keyValue) error {
	iterk := me.kpool.Get().(*reflect.Value)
	defer func() {
		iterk.SetZero()
		me.kpool.Put(iterk)
	}()
	iterv := me.vpool.Get().(*reflect.Value)
	defer func() {
		iterv.SetZero()
		me.vpool.Put(iterv)
	}()

	if kvs == nil {
		for i, iter := 0, v.MapRange(); iter.Next(); i++ {
			iterk.SetIterKey(iter)
			iterv.SetIterValue(iter)

			if err := me.kf(e, em, *iterk); err != nil {
				return err
			}
			if err := me.ef(e, em, *iterv); err != nil {
				return err
			}
		}
		return nil
	}

	initial := e.Len()
	for i, iter := 0, v.MapRange(); iter.Next(); i++ {
		iterk.SetIterKey(iter)
		iterv.SetIterValue(iter)

		offset := e.Len()
		if err := me.kf(e, em, *iterk); err != nil {
			return err
		}
		valueOffset := e.Len()
		if err := me.ef(e, em, *iterv); err != nil {
			return err
		}
		kvs[i] = keyValue{
			offset:      offset - initial,
			valueOffset: valueOffset - initial,
			nextOffset:  e.Len() - initial,
		}
	}

	return nil
}

func getEncodeMapFunc(t reflect.Type) encodeFunc {
	kf, _, _ := getEncodeFunc(t.Key())
	ef, _, _ := getEncodeFunc(t.Elem())
	if kf == nil || ef == nil {
		return nil
	}
	mkv := &mapKeyValueEncodeFunc{
		kf: kf,
		ef: ef,
		kpool: sync.Pool{
			New: func() any {
				rk := reflect.New(t.Key()).Elem()
				return &rk
			},
		},
		vpool: sync.Pool{
			New: func() any {
				rv := reflect.New(t.Elem()).Elem()
				return &rv
			},
		},
	}
	return mapEncodeFunc{
		e: mkv.encodeKeyValues,
	}.encode
}
