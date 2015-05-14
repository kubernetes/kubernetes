// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
)

// Various errors the driver might return. Can change between driver versions.
var (
	ErrInvalidConn       = errors.New("Invalid Connection")
	ErrMalformPkt        = errors.New("Malformed Packet")
	ErrNoTLS             = errors.New("TLS encryption requested but server does not support TLS")
	ErrOldPassword       = errors.New("This user requires old password authentication. If you still want to use it, please add 'allowOldPasswords=1' to your DSN. See also https://github.com/go-sql-driver/mysql/wiki/old_passwords")
	ErrCleartextPassword = errors.New("This user requires clear text authentication. If you still want to use it, please add 'allowCleartextPasswords=1' to your DSN.")
	ErrUnknownPlugin     = errors.New("The authentication plugin is not supported.")
	ErrOldProtocol       = errors.New("MySQL-Server does not support required Protocol 41+")
	ErrPktSync           = errors.New("Commands out of sync. You can't run this command now")
	ErrPktSyncMul        = errors.New("Commands out of sync. Did you run multiple statements at once?")
	ErrPktTooLarge       = errors.New("Packet for query is too large. You can change this value on the server by adjusting the 'max_allowed_packet' variable.")
	ErrBusyBuffer        = errors.New("Busy buffer")
)

var errLog Logger = log.New(os.Stderr, "[MySQL] ", log.Ldate|log.Ltime|log.Lshortfile)

// Logger is used to log critical error messages.
type Logger interface {
	Print(v ...interface{})
}

// SetLogger is used to set the logger for critical errors.
// The initial logger is os.Stderr.
func SetLogger(logger Logger) error {
	if logger == nil {
		return errors.New("logger is nil")
	}
	errLog = logger
	return nil
}

// MySQLError is an error type which represents a single MySQL error
type MySQLError struct {
	Number  uint16
	Message string
}

func (me *MySQLError) Error() string {
	return fmt.Sprintf("Error %d: %s", me.Number, me.Message)
}

// MySQLWarnings is an error type which represents a group of one or more MySQL
// warnings
type MySQLWarnings []MySQLWarning

func (mws MySQLWarnings) Error() string {
	var msg string
	for i, warning := range mws {
		if i > 0 {
			msg += "\r\n"
		}
		msg += fmt.Sprintf(
			"%s %s: %s",
			warning.Level,
			warning.Code,
			warning.Message,
		)
	}
	return msg
}

// MySQLWarning is an error type which represents a single MySQL warning.
// Warnings are returned in groups only. See MySQLWarnings
type MySQLWarning struct {
	Level   string
	Code    string
	Message string
}

func (mc *mysqlConn) getWarnings() (err error) {
	rows, err := mc.Query("SHOW WARNINGS", nil)
	if err != nil {
		return
	}

	var warnings = MySQLWarnings{}
	var values = make([]driver.Value, 3)

	for {
		err = rows.Next(values)
		switch err {
		case nil:
			warning := MySQLWarning{}

			if raw, ok := values[0].([]byte); ok {
				warning.Level = string(raw)
			} else {
				warning.Level = fmt.Sprintf("%s", values[0])
			}
			if raw, ok := values[1].([]byte); ok {
				warning.Code = string(raw)
			} else {
				warning.Code = fmt.Sprintf("%s", values[1])
			}
			if raw, ok := values[2].([]byte); ok {
				warning.Message = string(raw)
			} else {
				warning.Message = fmt.Sprintf("%s", values[0])
			}

			warnings = append(warnings, warning)

		case io.EOF:
			return warnings

		default:
			rows.Close()
			return
		}
	}
}
