// Copyright (C) 2014 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.
// +build icu

package sqlite3

/*
#cgo LDFLAGS: -licuuc -licui18n
#cgo CFLAGS: -DSQLITE_ENABLE_ICU
*/
import "C"
