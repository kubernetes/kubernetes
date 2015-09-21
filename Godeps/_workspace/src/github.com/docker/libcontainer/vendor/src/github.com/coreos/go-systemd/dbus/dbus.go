/*
Copyright 2013 CoreOS Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Integration with the systemd D-Bus API.  See http://www.freedesktop.org/wiki/Software/systemd/dbus/
package dbus

import (
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/godbus/dbus"
)

const signalBuffer = 100

// ObjectPath creates a dbus.ObjectPath using the rules that systemd uses for
// serializing special characters.
func ObjectPath(path string) dbus.ObjectPath {
	path = strings.Replace(path, ".", "_2e", -1)
	path = strings.Replace(path, "-", "_2d", -1)
	path = strings.Replace(path, "@", "_40", -1)

	return dbus.ObjectPath(path)
}

// Conn is a connection to systemds dbus endpoint.
type Conn struct {
	sysconn     *dbus.Conn
	sysobj      *dbus.Object
	jobListener struct {
		jobs map[dbus.ObjectPath]chan string
		sync.Mutex
	}
	subscriber struct {
		updateCh chan<- *SubStateUpdate
		errCh    chan<- error
		sync.Mutex
		ignore      map[dbus.ObjectPath]int64
		cleanIgnore int64
	}
	dispatch map[string]func(dbus.Signal)
}

// New() establishes a connection to the system bus and authenticates.
func New() (*Conn, error) {
	c := new(Conn)

	if err := c.initConnection(); err != nil {
		return nil, err
	}

	c.initJobs()
	return c, nil
}

func (c *Conn) initConnection() error {
	var err error
	c.sysconn, err = dbus.SystemBusPrivate()
	if err != nil {
		return err
	}

	// Only use EXTERNAL method, and hardcode the uid (not username)
	// to avoid a username lookup (which requires a dynamically linked
	// libc)
	methods := []dbus.Auth{dbus.AuthExternal(strconv.Itoa(os.Getuid()))}

	err = c.sysconn.Auth(methods)
	if err != nil {
		c.sysconn.Close()
		return err
	}

	err = c.sysconn.Hello()
	if err != nil {
		c.sysconn.Close()
		return err
	}

	c.sysobj = c.sysconn.Object("org.freedesktop.systemd1", dbus.ObjectPath("/org/freedesktop/systemd1"))

	// Setup the listeners on jobs so that we can get completions
	c.sysconn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
		"type='signal', interface='org.freedesktop.systemd1.Manager', member='JobRemoved'")
	c.initSubscription()
	c.initDispatch()

	return nil
}
