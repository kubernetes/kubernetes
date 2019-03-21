// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package journal provides write bindings to the local systemd journal.
// It is implemented in pure Go and connects to the journal directly over its
// unix socket.
//
// To read from the journal, see the "sdjournal" package, which wraps the
// sd-journal a C API.
//
// http://www.freedesktop.org/software/systemd/man/systemd-journald.service.html
package journal

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// Priority of a journal message
type Priority int

const (
	PriEmerg Priority = iota
	PriAlert
	PriCrit
	PriErr
	PriWarning
	PriNotice
	PriInfo
	PriDebug
)

var conn net.Conn

func init() {
	var err error
	conn, err = net.Dial("unixgram", "/run/systemd/journal/socket")
	if err != nil {
		conn = nil
	}
}

// Enabled returns true if the local systemd journal is available for logging
func Enabled() bool {
	return conn != nil
}

// Send a message to the local systemd journal. vars is a map of journald
// fields to values.  Fields must be composed of uppercase letters, numbers,
// and underscores, but must not start with an underscore. Within these
// restrictions, any arbitrary field name may be used.  Some names have special
// significance: see the journalctl documentation
// (http://www.freedesktop.org/software/systemd/man/systemd.journal-fields.html)
// for more details.  vars may be nil.
func Send(message string, priority Priority, vars map[string]string) error {
	if conn == nil {
		return journalError("could not connect to journald socket")
	}

	data := new(bytes.Buffer)
	appendVariable(data, "PRIORITY", strconv.Itoa(int(priority)))
	appendVariable(data, "MESSAGE", message)
	for k, v := range vars {
		appendVariable(data, k, v)
	}

	_, err := io.Copy(conn, data)
	if err != nil && isSocketSpaceError(err) {
		file, err := tempFd()
		if err != nil {
			return journalError(err.Error())
		}
		defer file.Close()
		_, err = io.Copy(file, data)
		if err != nil {
			return journalError(err.Error())
		}

		rights := syscall.UnixRights(int(file.Fd()))

		/* this connection should always be a UnixConn, but better safe than sorry */
		unixConn, ok := conn.(*net.UnixConn)
		if !ok {
			return journalError("can't send file through non-Unix connection")
		}
		_, _, err = unixConn.WriteMsgUnix([]byte{}, rights, nil)
		if err != nil {
			return journalError(err.Error())
		}
	} else if err != nil {
		return journalError(err.Error())
	}
	return nil
}

// Print prints a message to the local systemd journal using Send().
func Print(priority Priority, format string, a ...interface{}) error {
	return Send(fmt.Sprintf(format, a...), priority, nil)
}

func appendVariable(w io.Writer, name, value string) {
	if !validVarName(name) {
		journalError("variable name contains invalid character, ignoring")
	}
	if strings.ContainsRune(value, '\n') {
		/* When the value contains a newline, we write:
		 * - the variable name, followed by a newline
		 * - the size (in 64bit little endian format)
		 * - the data, followed by a newline
		 */
		fmt.Fprintln(w, name)
		binary.Write(w, binary.LittleEndian, uint64(len(value)))
		fmt.Fprintln(w, value)
	} else {
		/* just write the variable and value all on one line */
		fmt.Fprintf(w, "%s=%s\n", name, value)
	}
}

func validVarName(name string) bool {
	/* The variable name must be in uppercase and consist only of characters,
	 * numbers and underscores, and may not begin with an underscore. (from the docs)
	 */

	valid := name[0] != '_'
	for _, c := range name {
		valid = valid && ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '_'
	}
	return valid
}

func isSocketSpaceError(err error) bool {
	opErr, ok := err.(*net.OpError)
	if !ok {
		return false
	}

	sysErr, ok := opErr.Err.(syscall.Errno)
	if !ok {
		return false
	}

	return sysErr == syscall.EMSGSIZE || sysErr == syscall.ENOBUFS
}

func tempFd() (*os.File, error) {
	file, err := ioutil.TempFile("/dev/shm/", "journal.XXXXX")
	if err != nil {
		return nil, err
	}
	err = syscall.Unlink(file.Name())
	if err != nil {
		return nil, err
	}
	return file, nil
}

func journalError(s string) error {
	s = "journal error: " + s
	fmt.Fprintln(os.Stderr, s)
	return errors.New(s)
}
