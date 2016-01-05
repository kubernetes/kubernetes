// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"fmt"
	"io"
	"os"
	"strings"
)

var (
	fileRegister   map[string]bool
	readerRegister map[string]func() io.Reader
)

// RegisterLocalFile adds the given file to the file whitelist,
// so that it can be used by "LOAD DATA LOCAL INFILE <filepath>".
// Alternatively you can allow the use of all local files with
// the DSN parameter 'allowAllFiles=true'
//
//  filePath := "/home/gopher/data.csv"
//  mysql.RegisterLocalFile(filePath)
//  err := db.Exec("LOAD DATA LOCAL INFILE '" + filePath + "' INTO TABLE foo")
//  if err != nil {
//  ...
//
func RegisterLocalFile(filePath string) {
	// lazy map init
	if fileRegister == nil {
		fileRegister = make(map[string]bool)
	}

	fileRegister[strings.Trim(filePath, `"`)] = true
}

// DeregisterLocalFile removes the given filepath from the whitelist.
func DeregisterLocalFile(filePath string) {
	delete(fileRegister, strings.Trim(filePath, `"`))
}

// RegisterReaderHandler registers a handler function which is used
// to receive a io.Reader.
// The Reader can be used by "LOAD DATA LOCAL INFILE Reader::<name>".
// If the handler returns a io.ReadCloser Close() is called when the
// request is finished.
//
//  mysql.RegisterReaderHandler("data", func() io.Reader {
//  	var csvReader io.Reader // Some Reader that returns CSV data
//  	... // Open Reader here
//  	return csvReader
//  })
//  err := db.Exec("LOAD DATA LOCAL INFILE 'Reader::data' INTO TABLE foo")
//  if err != nil {
//  ...
//
func RegisterReaderHandler(name string, handler func() io.Reader) {
	// lazy map init
	if readerRegister == nil {
		readerRegister = make(map[string]func() io.Reader)
	}

	readerRegister[name] = handler
}

// DeregisterReaderHandler removes the ReaderHandler function with
// the given name from the registry.
func DeregisterReaderHandler(name string) {
	delete(readerRegister, name)
}

func deferredClose(err *error, closer io.Closer) {
	closeErr := closer.Close()
	if *err == nil {
		*err = closeErr
	}
}

func (mc *mysqlConn) handleInFileRequest(name string) (err error) {
	var rdr io.Reader
	var data []byte

	if idx := strings.Index(name, "Reader::"); idx == 0 || (idx > 0 && name[idx-1] == '/') { // io.Reader
		// The server might return an an absolute path. See issue #355.
		name = name[idx+8:]

		if handler, inMap := readerRegister[name]; inMap {
			rdr = handler()
			if rdr != nil {
				data = make([]byte, 4+mc.maxWriteSize)

				if cl, ok := rdr.(io.Closer); ok {
					defer deferredClose(&err, cl)
				}
			} else {
				err = fmt.Errorf("Reader '%s' is <nil>", name)
			}
		} else {
			err = fmt.Errorf("Reader '%s' is not registered", name)
		}
	} else { // File
		name = strings.Trim(name, `"`)
		if mc.cfg.allowAllFiles || fileRegister[name] {
			var file *os.File
			var fi os.FileInfo

			if file, err = os.Open(name); err == nil {
				defer deferredClose(&err, file)

				// get file size
				if fi, err = file.Stat(); err == nil {
					rdr = file
					if fileSize := int(fi.Size()); fileSize <= mc.maxWriteSize {
						data = make([]byte, 4+fileSize)
					} else if fileSize <= mc.maxPacketAllowed {
						data = make([]byte, 4+mc.maxWriteSize)
					} else {
						err = fmt.Errorf("Local File '%s' too large: Size: %d, Max: %d", name, fileSize, mc.maxPacketAllowed)
					}
				}
			}
		} else {
			err = fmt.Errorf("Local File '%s' is not registered. Use the DSN parameter 'allowAllFiles=true' to allow all files", name)
		}
	}

	// send content packets
	if err == nil {
		var n int
		for err == nil {
			n, err = rdr.Read(data[4:])
			if n > 0 {
				if ioErr := mc.writePacket(data[:4+n]); ioErr != nil {
					return ioErr
				}
			}
		}
		if err == io.EOF {
			err = nil
		}
	}

	// send empty packet (termination)
	if data == nil {
		data = make([]byte, 4)
	}
	if ioErr := mc.writePacket(data[:4]); ioErr != nil {
		return ioErr
	}

	// read OK packet
	if err == nil {
		return mc.readResultOK()
	} else {
		mc.readPacket()
	}
	return err
}
