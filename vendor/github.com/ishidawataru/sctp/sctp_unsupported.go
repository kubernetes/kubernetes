// +build !linux linux,386
// Copyright 2019 Wataru Ishida. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sctp

import (
	"errors"
	"net"
	"runtime"
	"syscall"
)

var ErrUnsupported = errors.New("SCTP is unsupported on " + runtime.GOOS + "/" + runtime.GOARCH)

func setsockopt(fd int, optname, optval, optlen uintptr) (uintptr, uintptr, error) {
	return 0, 0, ErrUnsupported
}

func getsockopt(fd int, optname, optval, optlen uintptr) (uintptr, uintptr, error) {
	return 0, 0, ErrUnsupported
}

func (c *SCTPConn) SCTPWrite(b []byte, info *SndRcvInfo) (int, error) {
	return 0, ErrUnsupported
}

func (c *SCTPConn) SCTPRead(b []byte) (int, *SndRcvInfo, error) {
	return 0, nil, ErrUnsupported
}

func (c *SCTPConn) Close() error {
	return ErrUnsupported
}

func (c *SCTPConn) SetWriteBuffer(bytes int) error {
	return ErrUnsupported
}

func (c *SCTPConn) GetWriteBuffer() (int, error) {
	return 0, ErrUnsupported
}

func (c *SCTPConn) SetReadBuffer(bytes int) error {
	return ErrUnsupported
}

func (c *SCTPConn) GetReadBuffer() (int, error) {
	return 0, ErrUnsupported
}

func ListenSCTP(net string, laddr *SCTPAddr) (*SCTPListener, error) {
	return nil, ErrUnsupported
}

func ListenSCTPExt(net string, laddr *SCTPAddr, options InitMsg) (*SCTPListener, error) {
	return nil, ErrUnsupported
}

func listenSCTPExtConfig(network string, laddr *SCTPAddr, options InitMsg, control func(network, address string, c syscall.RawConn) error) (*SCTPListener, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) Accept() (net.Conn, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) AcceptSCTP() (*SCTPConn, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) Close() error {
	return ErrUnsupported
}

func DialSCTP(net string, laddr, raddr *SCTPAddr) (*SCTPConn, error) {
	return nil, ErrUnsupported
}

func DialSCTPExt(network string, laddr, raddr *SCTPAddr, options InitMsg) (*SCTPConn, error) {
	return nil, ErrUnsupported
}

func dialSCTPExtConfig(network string, laddr, raddr *SCTPAddr, options InitMsg, control func(network, address string, c syscall.RawConn) error) (*SCTPConn, error) {
	return nil, ErrUnsupported
}
