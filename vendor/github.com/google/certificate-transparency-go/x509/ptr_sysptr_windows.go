// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11

package x509

import (
	"syscall"
	"unsafe"
)

// For Go versions >= 1.11, the ExtraPolicyPara field in
// syscall.CertChainPolicyPara is of type syscall.Pointer.  See:
//   https://github.com/golang/go/commit/4869ec00e87ef

func convertToPolicyParaType(p unsafe.Pointer) syscall.Pointer {
	return (syscall.Pointer)(p)
}
