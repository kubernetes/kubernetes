// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build go1.5

package codec

import "reflect"

const reflectArrayOfSupported = true

func reflectArrayOf(rvn reflect.Value) (rvn2 reflect.Value) {
	rvn2 = reflect.New(reflect.ArrayOf(rvn.Len(), intfTyp)).Elem()
	reflect.Copy(rvn2, rvn)
	return
}
