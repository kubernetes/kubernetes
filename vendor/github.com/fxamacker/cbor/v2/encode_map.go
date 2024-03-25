// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

//go:build go1.20

package cbor

import (
	"reflect"
	"sync"
)

type mapKeyValueEncodeFunc struct {
	kf, ef       encodeFunc
	kpool, vpool sync.Pool
}

func (me *mapKeyValueEncodeFunc) encodeKeyValues(e *encoderBuffer, em *encMode, v reflect.Value, kvs []keyValue) error {
	trackKeyValueLength := len(kvs) == v.Len()
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
	iter := v.MapRange()
	for i := 0; iter.Next(); i++ {
		off := e.Len()
		iterk.SetIterKey(iter)
		iterv.SetIterValue(iter)

		if err := me.kf(e, em, *iterk); err != nil {
			return err
		}
		if trackKeyValueLength {
			kvs[i].keyLen = e.Len() - off
		}

		if err := me.ef(e, em, *iterv); err != nil {
			return err
		}
		if trackKeyValueLength {
			kvs[i].keyValueLen = e.Len() - off
		}
	}

	return nil
}

func getEncodeMapFunc(t reflect.Type) encodeFunc {
	kf, _ := getEncodeFunc(t.Key())
	ef, _ := getEncodeFunc(t.Elem())
	if kf == nil || ef == nil {
		return nil
	}
	mkv := &mapKeyValueEncodeFunc{
		kf: kf,
		ef: ef,
		kpool: sync.Pool{
			New: func() interface{} {
				rk := reflect.New(t.Key()).Elem()
				return &rk
			},
		},
		vpool: sync.Pool{
			New: func() interface{} {
				rv := reflect.New(t.Elem()).Elem()
				return &rv
			},
		},
	}
	return mapEncodeFunc{
		e: mkv.encodeKeyValues,
	}.encode
}
