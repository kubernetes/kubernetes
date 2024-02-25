// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

//go:build !go1.20

package cbor

import (
	"reflect"
)

type mapKeyValueEncodeFunc struct {
	kf, ef encodeFunc
}

func (me *mapKeyValueEncodeFunc) encodeKeyValues(e *encoderBuffer, em *encMode, v reflect.Value, kvs []keyValue) error {
	trackKeyValueLength := len(kvs) == v.Len()

	iter := v.MapRange()
	for i := 0; iter.Next(); i++ {
		off := e.Len()

		if err := me.kf(e, em, iter.Key()); err != nil {
			return err
		}
		if trackKeyValueLength {
			kvs[i].keyLen = e.Len() - off
		}

		if err := me.ef(e, em, iter.Value()); err != nil {
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
	mkv := &mapKeyValueEncodeFunc{kf: kf, ef: ef}
	return mapEncodeFunc{
		e: mkv.encodeKeyValues,
	}.encode
}
