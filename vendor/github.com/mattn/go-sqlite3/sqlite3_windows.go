// Copyright (C) 2014 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.
// +build windows

package sqlite3

/*
#cgo CFLAGS: -I. -fno-stack-check -fno-stack-protector -mno-stack-arg-probe
#cgo windows,386 CFLAGS: -D_USE_32BIT_TIME_T
#cgo LDFLAGS: -lmingwex -lmingw32
*/
import "C"
