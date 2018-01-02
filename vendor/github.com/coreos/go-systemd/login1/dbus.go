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

// Integration with the systemd logind API.  See http://www.freedesktop.org/wiki/Software/systemd/logind/
package login1

import (
	"fmt"
	"os"
	"strconv"

	"github.com/godbus/dbus"
)

const (
	dbusInterface = "org.freedesktop.login1.Manager"
	dbusPath      = "/org/freedesktop/login1"
)

// Conn is a connection to systemds dbus endpoint.
type Conn struct {
	conn   *dbus.Conn
	object dbus.BusObject
}

// New() establishes a connection to the system bus and authenticates.
func New() (*Conn, error) {
	c := new(Conn)

	if err := c.initConnection(); err != nil {
		return nil, err
	}

	return c, nil
}

func (c *Conn) initConnection() error {
	var err error
	c.conn, err = dbus.SystemBusPrivate()
	if err != nil {
		return err
	}

	// Only use EXTERNAL method, and hardcode the uid (not username)
	// to avoid a username lookup (which requires a dynamically linked
	// libc)
	methods := []dbus.Auth{dbus.AuthExternal(strconv.Itoa(os.Getuid()))}

	err = c.conn.Auth(methods)
	if err != nil {
		c.conn.Close()
		return err
	}

	err = c.conn.Hello()
	if err != nil {
		c.conn.Close()
		return err
	}

	c.object = c.conn.Object("org.freedesktop.login1", dbus.ObjectPath(dbusPath))

	return nil
}

// Reboot asks logind for a reboot optionally asking for auth.
func (c *Conn) Reboot(askForAuth bool) {
	c.object.Call(dbusInterface+".Reboot", 0, askForAuth)
}

// Inhibit takes inhibition lock in logind.
func (c *Conn) Inhibit(what, who, why, mode string) (*os.File, error) {
	var fd dbus.UnixFD

	err := c.object.Call(dbusInterface+".Inhibit", 0, what, who, why, mode).Store(&fd)
	if err != nil {
		return nil, err
	}

	return os.NewFile(uintptr(fd), "inhibit"), nil
}

// Subscribe to signals on the logind dbus
func (c *Conn) Subscribe(members ...string) chan *dbus.Signal {
	for _, member := range members {
		c.conn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
			fmt.Sprintf("type='signal',interface='org.freedesktop.login1.Manager',member='%s'", member))
	}
	ch := make(chan *dbus.Signal, 10)
	c.conn.Signal(ch)
	return ch
}

// PowerOff asks logind for a power off optionally asking for auth.
func (c *Conn) PowerOff(askForAuth bool) {
	c.object.Call(dbusInterface+".PowerOff", 0, askForAuth)
}
