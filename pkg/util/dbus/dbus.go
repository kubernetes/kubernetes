/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package dbus

import (
	godbus "github.com/godbus/dbus"
)

// Interface is an interface that presents a subset of the godbus/dbus API.  Use this
// when you want to inject fakeable/mockable D-Bus behavior.
type Interface interface {
	// SystemBus returns a connection to the system bus, connecting to it
	// first if necessary
	SystemBus() (Connection, error)
	// SessionBus returns a connection to the session bus, connecting to it
	// first if necessary
	SessionBus() (Connection, error)
}

// Connection represents a D-Bus connection
type Connection interface {
	// Returns an Object representing the bus itself
	BusObject() Object

	// Object creates a representation of a remote D-Bus object
	Object(name, path string) Object

	// Signal registers or unregisters a channel to receive D-Bus signals
	Signal(ch chan<- *godbus.Signal)
}

// Object represents a remote D-Bus object
type Object interface {
	// Call synchronously calls a D-Bus method
	Call(method string, flags godbus.Flags, args ...interface{}) Call
}

// Call represents a pending or completed D-Bus method call
type Call interface {
	// Store returns a completed call's return values, or an error
	Store(retvalues ...interface{}) error
}

// Implements Interface in terms of actually talking to D-Bus
type dbusImpl struct {
	systemBus  *connImpl
	sessionBus *connImpl
}

// Implements Connection as a godbus.Conn
type connImpl struct {
	conn *godbus.Conn
}

// Implements Object as a godbus.Object
type objectImpl struct {
	object *godbus.Object
}

// Implements Call as a godbus.Call
type callImpl struct {
	call *godbus.Call
}

// New returns a new Interface which will use godbus to talk to D-Bus
func New() Interface {
	return &dbusImpl{}
}

// SystemBus is part of Interface
func (db *dbusImpl) SystemBus() (Connection, error) {
	if db.systemBus == nil {
		bus, err := godbus.SystemBus()
		if err != nil {
			return nil, err
		}
		db.systemBus = &connImpl{bus}
	}

	return db.systemBus, nil
}

// SessionBus is part of Interface
func (db *dbusImpl) SessionBus() (Connection, error) {
	if db.sessionBus == nil {
		bus, err := godbus.SessionBus()
		if err != nil {
			return nil, err
		}
		db.sessionBus = &connImpl{bus}
	}

	return db.sessionBus, nil
}

// BusObject is part of the Connection interface
func (conn *connImpl) BusObject() Object {
	return &objectImpl{conn.conn.BusObject()}
}

// Object is part of the Connection interface
func (conn *connImpl) Object(name, path string) Object {
	return &objectImpl{conn.conn.Object(name, godbus.ObjectPath(path))}
}

// Signal is part of the Connection interface
func (conn *connImpl) Signal(ch chan<- *godbus.Signal) {
	conn.conn.Signal(ch)
}

// Call is part of the Object interface
func (obj *objectImpl) Call(method string, flags godbus.Flags, args ...interface{}) Call {
	return &callImpl{obj.object.Call(method, flags, args...)}
}

// Store is part of the Call interface
func (call *callImpl) Store(retvalues ...interface{}) error {
	return call.call.Store(retvalues...)
}
