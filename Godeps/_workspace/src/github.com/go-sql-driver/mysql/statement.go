// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2012 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"database/sql/driver"
	"fmt"
	"reflect"
	"strconv"
)

type mysqlStmt struct {
	mc         *mysqlConn
	id         uint32
	paramCount int
	columns    []mysqlField // cached from the first query
}

func (stmt *mysqlStmt) Close() error {
	if stmt.mc == nil || stmt.mc.netConn == nil {
		errLog.Print(ErrInvalidConn)
		return driver.ErrBadConn
	}

	err := stmt.mc.writeCommandPacketUint32(comStmtClose, stmt.id)
	stmt.mc = nil
	return err
}

func (stmt *mysqlStmt) NumInput() int {
	return stmt.paramCount
}

func (stmt *mysqlStmt) ColumnConverter(idx int) driver.ValueConverter {
	return converter{}
}

func (stmt *mysqlStmt) Exec(args []driver.Value) (driver.Result, error) {
	if stmt.mc.netConn == nil {
		errLog.Print(ErrInvalidConn)
		return nil, driver.ErrBadConn
	}
	// Send command
	err := stmt.writeExecutePacket(args)
	if err != nil {
		return nil, err
	}

	mc := stmt.mc

	mc.affectedRows = 0
	mc.insertId = 0

	// Read Result
	resLen, err := mc.readResultSetHeaderPacket()
	if err == nil {
		if resLen > 0 {
			// Columns
			err = mc.readUntilEOF()
			if err != nil {
				return nil, err
			}

			// Rows
			err = mc.readUntilEOF()
		}
		if err == nil {
			return &mysqlResult{
				affectedRows: int64(mc.affectedRows),
				insertId:     int64(mc.insertId),
			}, nil
		}
	}

	return nil, err
}

func (stmt *mysqlStmt) Query(args []driver.Value) (driver.Rows, error) {
	if stmt.mc.netConn == nil {
		errLog.Print(ErrInvalidConn)
		return nil, driver.ErrBadConn
	}
	// Send command
	err := stmt.writeExecutePacket(args)
	if err != nil {
		return nil, err
	}

	mc := stmt.mc

	// Read Result
	resLen, err := mc.readResultSetHeaderPacket()
	if err != nil {
		return nil, err
	}

	rows := new(binaryRows)
	rows.mc = mc

	if resLen > 0 {
		// Columns
		// If not cached, read them and cache them
		if stmt.columns == nil {
			rows.columns, err = mc.readColumns(resLen)
			stmt.columns = rows.columns
		} else {
			rows.columns = stmt.columns
			err = mc.readUntilEOF()
		}
	}

	return rows, err
}

type converter struct{}

func (c converter) ConvertValue(v interface{}) (driver.Value, error) {
	if driver.IsValue(v) {
		return v, nil
	}

	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Ptr:
		// indirect pointers
		if rv.IsNil() {
			return nil, nil
		}
		return c.ConvertValue(rv.Elem().Interface())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32:
		return int64(rv.Uint()), nil
	case reflect.Uint64:
		u64 := rv.Uint()
		if u64 >= 1<<63 {
			return strconv.FormatUint(u64, 10), nil
		}
		return int64(u64), nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	}
	return nil, fmt.Errorf("unsupported type %T, a %s", v, rv.Kind())
}
