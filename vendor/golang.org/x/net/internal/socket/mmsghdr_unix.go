// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || linux || netbsd
// +build aix linux netbsd

package socket

import "net"

type mmsghdrs []mmsghdr

func (hs mmsghdrs) pack(ms []Message, parseFn func([]byte, string) (net.Addr, error), marshalFn func(net.Addr) []byte) error {
	for i := range hs {
		vs := make([]iovec, len(ms[i].Buffers))
		var sa []byte
		if parseFn != nil {
			sa = make([]byte, sizeofSockaddrInet6)
		}
		if marshalFn != nil {
			sa = marshalFn(ms[i].Addr)
		}
		hs[i].Hdr.pack(vs, ms[i].Buffers, ms[i].OOB, sa)
	}
	return nil
}

func (hs mmsghdrs) unpack(ms []Message, parseFn func([]byte, string) (net.Addr, error), hint string) error {
	for i := range hs {
		ms[i].N = int(hs[i].Len)
		ms[i].NN = hs[i].Hdr.controllen()
		ms[i].Flags = hs[i].Hdr.flags()
		if parseFn != nil {
			var err error
			ms[i].Addr, err = parseFn(hs[i].Hdr.name(), hint)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
