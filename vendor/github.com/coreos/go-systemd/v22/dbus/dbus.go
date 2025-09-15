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
	"context"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/godbus/dbus/v5"
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

// pathBusUnescape is the inverse of PathBusEscape.
func pathBusUnescape(path string) string {
	if path == "_" {
		return ""
	}
	n := []byte{}
	for i := 0; i < len(path); i++ {
		c := path[i]
		if c == '_' && i+2 < len(path) {
			res, err := hex.DecodeString(path[i+1 : i+3])
			if err == nil {
				n = append(n, res...)
			}
			i += 2
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
	sysobj  dbus.BusObject

	// sigconn/sigobj are only used to receive dbus signals
	sigconn *dbus.Conn
	sigobj  dbus.BusObject

	jobListener struct {
		jobs map[dbus.ObjectPath]chan<- string
		sync.Mutex
	}
	subStateSubscriber struct {
		updateCh chan<- *SubStateUpdate
		errCh    chan<- error
		sync.Mutex
		ignore      map[dbus.ObjectPath]int64
		cleanIgnore int64
	}
	propertiesSubscriber struct {
		updateCh chan<- *PropertiesUpdate
		errCh    chan<- error
		sync.Mutex
	}
}

// Deprecated: use NewWithContext instead.
func New() (*Conn, error) {
	return NewWithContext(context.Background())
}

// NewWithContext establishes a connection to any available bus and authenticates.
// Callers should call Close() when done with the connection.
func NewWithContext(ctx context.Context) (*Conn, error) {
	conn, err := NewSystemConnectionContext(ctx)
	if err != nil && os.Geteuid() == 0 {
		return NewSystemdConnectionContext(ctx)
	}
	return conn, err
}

// Deprecated: use NewSystemConnectionContext instead.
func NewSystemConnection() (*Conn, error) {
	return NewSystemConnectionContext(context.Background())
}

// NewSystemConnectionContext establishes a connection to the system bus and authenticates.
// Callers should call Close() when done with the connection.
func NewSystemConnectionContext(ctx context.Context) (*Conn, error) {
	return NewConnection(func() (*dbus.Conn, error) {
		return dbusAuthHelloConnection(ctx, dbus.SystemBusPrivate)
	})
}

// Deprecated: use NewUserConnectionContext instead.
func NewUserConnection() (*Conn, error) {
	return NewUserConnectionContext(context.Background())
}

// NewUserConnectionContext establishes a connection to the session bus and
// authenticates. This can be used to connect to systemd user instances.
// Callers should call Close() when done with the connection.
func NewUserConnectionContext(ctx context.Context) (*Conn, error) {
	return NewConnection(func() (*dbus.Conn, error) {
		return dbusAuthHelloConnection(ctx, dbus.SessionBusPrivate)
	})
}

// Deprecated: use NewSystemdConnectionContext instead.
func NewSystemdConnection() (*Conn, error) {
	return NewSystemdConnectionContext(context.Background())
}

// NewSystemdConnectionContext establishes a private, direct connection to systemd.
// This can be used for communicating with systemd without a dbus daemon.
// Callers should call Close() when done with the connection.
func NewSystemdConnectionContext(ctx context.Context) (*Conn, error) {
	return NewConnection(func() (*dbus.Conn, error) {
		// We skip Hello when talking directly to systemd.
		return dbusAuthConnection(ctx, func(opts ...dbus.ConnOption) (*dbus.Conn, error) {
			return dbus.Dial("unix:path=/run/systemd/private", opts...)
		})
	})
}

// Close closes an established connection.
func (c *Conn) Close() {
	c.sysconn.Close()
	c.sigconn.Close()
}

// Connected returns whether conn is connected
func (c *Conn) Connected() bool {
	return c.sysconn.Connected() && c.sigconn.Connected()
}

// NewConnection establishes a connection to a bus using a caller-supplied function.
// This allows connecting to remote buses through a user-supplied mechanism.
// The supplied function may be called multiple times, and should return independent connections.
// The returned connection must be fully initialised: the org.freedesktop.DBus.Hello call must have succeeded,
// and any authentication should be handled by the function.
func NewConnection(dialBus func() (*dbus.Conn, error)) (*Conn, error) {
	sysconn, err := dialBus()
	if err != nil {
		return nil, err
	}

	sigconn, err := dialBus()
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

	c.subStateSubscriber.ignore = make(map[dbus.ObjectPath]int64)
	c.jobListener.jobs = make(map[dbus.ObjectPath]chan<- string)

	// Setup the listeners on jobs so that we can get completions
	c.sigconn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
		"type='signal', interface='org.freedesktop.systemd1.Manager', member='JobRemoved'")

	c.dispatch()
	return c, nil
}

// GetManagerProperty returns the value of a property on the org.freedesktop.systemd1.Manager
// interface. The value is returned in its string representation, as defined at
// https://developer.gnome.org/glib/unstable/gvariant-text.html.
func (c *Conn) GetManagerProperty(prop string) (string, error) {
	variant, err := c.sysobj.GetProperty("org.freedesktop.systemd1.Manager." + prop)
	if err != nil {
		return "", err
	}
	return variant.String(), nil
}

func dbusAuthConnection(ctx context.Context, createBus func(opts ...dbus.ConnOption) (*dbus.Conn, error)) (*dbus.Conn, error) {
	conn, err := createBus(dbus.WithContext(ctx))
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

	return conn, nil
}

func dbusAuthHelloConnection(ctx context.Context, createBus func(opts ...dbus.ConnOption) (*dbus.Conn, error)) (*dbus.Conn, error) {
	conn, err := dbusAuthConnection(ctx, createBus)
	if err != nil {
		return nil, err
	}

	if err = conn.Hello(); err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}

func systemdObject(conn *dbus.Conn) dbus.BusObject {
	return conn.Object("org.freedesktop.systemd1", dbus.ObjectPath("/org/freedesktop/systemd1"))
}
