// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build go1.5

package codec

import "reflect"

const reflectArrayOfSupported = true

func reflectArrayOf(count int, elem reflect.Type) reflect.Type {
	return reflect.ArrayOf(count, elem)
}
