// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build !go1.4

package codec

// This codec package will only work for go1.4 and above.
// This is for the following reasons:
//   - go 1.4 was released in 2014
//   - go runtime is written fully in go
//   - interface only holds pointers
//   - reflect.Value is stabilized as 3 words

func init() {
	panic("codec: go 1.3 and below are not supported")
}
