// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

func (h *msghdr) setIov(vs []iovec) {
	h.Iov = &vs[0]
	h.Iovlen = uint32(len(vs))
}
