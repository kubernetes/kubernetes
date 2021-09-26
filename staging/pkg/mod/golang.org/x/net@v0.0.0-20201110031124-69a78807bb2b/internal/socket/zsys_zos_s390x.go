// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

type iovec struct {
	Base *byte
	Len  uint64
}

type msghdr struct {
	Name       *byte
	Iov        *iovec
	Control    *byte
	Flags      int32
	Namelen    uint32
	Iovlen     int32
	Controllen uint32
}

type cmsghdr struct {
	Len   int32
	Level int32
	Type  int32
}

const (
	sizeofCmsghdr       = 12
	sizeofSockaddrInet  = 16
	sizeofSockaddrInet6 = 28
)
