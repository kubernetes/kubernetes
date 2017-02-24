// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build !go1.5

package codec

import "reflect"

const reflectArrayOfSupported = false

func reflectArrayOf(rvn reflect.Value) (rvn2 reflect.Value) {
	panic("reflect.ArrayOf unsupported")
}
