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

// Integration with the systemd D-Bus API.  See http://www.freedesktop.org/wiki/Software/systemd/dbus/
package dbus

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/godbus/dbus"
)

const (
	alpha        = `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`
	num          = `0123456789`
	alphanum     = alpha + num
	signalBuffer = 100
)

// needsEscape checks whether a byte in a potential dbus ObjectPath needs to be escaped
func needsEscape(i int, b byte) bool {
	// Escape everything that is not a-z-A-Z-0-9
	// Also escape 0-9 if it's the first character
	return strings.IndexByte(alphanum, b) == -1 ||
		(i == 0 && strings.IndexByte(num, b) != -1)
}

// PathBusEscape sanitizes a constituent string of a dbus ObjectPath using the
// rules that systemd uses for serializing special characters.
func PathBusEscape(path string) string {
	// Special case the empty string
	if len(path) == 0 {
		return "_"
	}
	n := []byte{}
	for i := 0; i < len(path); i++ {
		c := path[i]
		if needsEscape(i, c) {
			e := fmt.Sprintf("_%x", c)
			n = append(n, []byte(e)...)
		} else {
			n = append(n, c)
		}
	}
	return string(n)
}

// Conn is a connection to systemd's dbus endpoint.
type Conn struct {
	// sysconn/sysobj are only used to call dbus methods
	sysconn *dbus.Conn
	sysobj  *dbus.Object

	// sigconn/sigobj are only used to receive dbus signals
	sigconn *dbus.Conn
	sigobj  *dbus.Object

	jobListener struct {
		jobs map[dbus.ObjectPath]chan<- string
		sync.Mutex
	}
	subscriber struct {
		updateCh chan<- *SubStateUpdate
		errCh    chan<- error
		sync.Mutex
		ignore      map[dbus.ObjectPath]int64
		cleanIgnore int64
	}
}

// New establishes a connection to the system bus and authenticates.
// Callers should call Close() when done with the connection.
func New() (*Conn, error) {
	return newConnection(dbus.SystemBusPrivate)
}

// NewUserConnection establishes a connection to the session bus and
// authenticates. This can be used to connect to systemd user instances.
// Callers should call Close() when done with the connection.
func NewUserConnection() (*Conn, error) {
	return newConnection(dbus.SessionBusPrivate)
}

// Close closes an established connection
func (c *Conn) Close() {
	c.sysconn.Close()
	c.sigconn.Close()
}

func newConnection(createBus func() (*dbus.Conn, error)) (*Conn, error) {
	sysconn, err := dbusConnection(createBus)
	if err != nil {
		return nil, err
	}

	sigconn, err := dbusConnection(createBus)
	if err != nil {
		sysconn.Close()
		return nil, err
	}

	c := &Conn{
		sysconn: sysconn,
		sysobj:  systemdObject(sysconn),
		sigconn: sigconn,
		sigobj:  systemdObject(sigconn),
	}

	c.subscriber.ignore = make(map[dbus.ObjectPath]int64)
	c.jobListener.jobs = make(map[dbus.ObjectPath]chan<- string)

	// Setup the listeners on jobs so that we can get completions
	c.sigconn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
		"type='signal', interface='org.freedesktop.systemd1.Manager', member='JobRemoved'")

	c.dispatch()
	return c, nil
}

func dbusConnection(createBus func() (*dbus.Conn, error)) (*dbus.Conn, error) {
	conn, err := createBus()
	if err != nil {
		return nil, err
	}

	// Only use EXTERNAL method, and hardcode the uid (not username)
	// to avoid a username lookup (which requires a dynamically linked
	// libc)
	methods := []dbus.Auth{dbus.AuthExternal(strconv.Itoa(os.Getuid()))}

	err = conn.Auth(methods)
	if err != nil {
		conn.Close()
		return nil, err
	}

	err = conn.Hello()
	if err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}

func systemdObject(conn *dbus.Conn) *dbus.Object {
	return conn.Object("org.freedesktop.systemd1", dbus.ObjectPath("/org/freedesktop/systemd1"))
}
