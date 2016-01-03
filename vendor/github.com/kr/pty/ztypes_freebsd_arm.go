// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs types_freebsd.go

package pty

const (
	_C_SPECNAMELEN = 0x3f
)

type fiodgnameArg struct {
	Len int32
	Buf *byte
}
