// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

//go:build !go1.20

package cbor

import (
	"bytes"
	"reflect"
)

type mapKeyValueEncodeFunc struct {
	kf, ef encodeFunc
}

func (me *mapKeyValueEncodeFunc) encodeKeyValues(e *bytes.Buffer, em *encMode, v reflect.Value, kvs []keyValue) error {
	if kvs == nil {
		for i, iter := 0, v.MapRange(); iter.Next(); i++ {
			if err := me.kf(e, em, iter.Key()); err != nil {
				return err
			}
			if err := me.ef(e, em, iter.Value()); err != nil {
				return err
			}
		}
		return nil
	}

	initial := e.Len()
	for i, iter := 0, v.MapRange(); iter.Next(); i++ {
		offset := e.Len()
		if err := me.kf(e, em, iter.Key()); err != nil {
			return err
		}
		valueOffset := e.Len()
		if err := me.ef(e, em, iter.Value()); err != nil {
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
	kf, _ := getEncodeFunc(t.Key())
	ef, _ := getEncodeFunc(t.Elem())
	if kf == nil || ef == nil {
		return nil
	}
	mkv := &mapKeyValueEncodeFunc{kf: kf, ef: ef}
	return mapEncodeFunc{
		e: mkv.encodeKeyValues,
	}.encode
}
