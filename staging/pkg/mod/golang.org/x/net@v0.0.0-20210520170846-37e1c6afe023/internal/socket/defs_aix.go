// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package socket

/*
#include <sys/socket.h>

#include <netinet/in.h>
*/
import "C"

type iovec C.struct_iovec

type msghdr C.struct_msghdr

type mmsghdr C.struct_mmsghdr

type cmsghdr C.struct_cmsghdr

const (
	sizeofIovec  = C.sizeof_struct_iovec
	sizeofMsghdr = C.sizeof_struct_msghdr
)
