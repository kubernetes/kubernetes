// Copyright (C) 2014 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.
// +build sqlite_omit_load_extension

package sqlite3

/*
#cgo CFLAGS: -DSQLITE_OMIT_LOAD_EXTENSION
*/
import "C"
import (
	"errors"
)

func (c *SQLiteConn) loadExtensions(extensions []string) error {
	return errors.New("Extensions have been disabled for static builds")
}

func (c *SQLiteConn) LoadExtension(lib string, entry string) error {
	return errors.New("Extensions have been disabled for static builds")
}
