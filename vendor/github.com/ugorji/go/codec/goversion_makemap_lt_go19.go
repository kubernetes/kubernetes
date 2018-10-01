// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build !go1.9

package codec

import "reflect"

func makeMapReflect(t reflect.Type, size int) reflect.Value {
	return reflect.MakeMap(t)
}
