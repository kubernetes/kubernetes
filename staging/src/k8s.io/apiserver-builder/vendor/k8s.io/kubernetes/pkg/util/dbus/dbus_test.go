/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"os"
	"testing"

	godbus "github.com/godbus/dbus"
)

const (
	DBusNameFlagAllowReplacement uint32 = 1 << (iota + 1)
	DBusNameFlagReplaceExisting
	DBusNameFlagDoNotQueue
)

const (
	DBusRequestNameReplyPrimaryOwner uint32 = iota + 1
	DBusRequestNameReplyInQueue
	DBusRequestNameReplyExists
	DBusRequestNameReplyAlreadyOwner
)

const (
	DBusReleaseNameReplyReleased uint32 = iota + 1
	DBusReleaseNameReplyNonExistent
	DBusReleaseNameReplyNotOwner
)

func doDBusTest(t *testing.T, dbus Interface, real bool) {
	bus, err := dbus.SystemBus()
	if err != nil {
		if !real {
			t.Errorf("dbus.SystemBus() failed with fake Interface")
		}
		t.Skipf("D-Bus is not running: %v", err)
	}
	busObj := bus.BusObject()

	id := ""
	err = busObj.Call("org.freedesktop.DBus.GetId", 0).Store(&id)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if len(id) == 0 {
		t.Errorf("expected non-empty Id, got \"\"")
	}

	// Switch to the session bus for the rest, since the system bus is more
	// locked down (and thus harder to trick into emitting signals).

	bus, err = dbus.SessionBus()
	if err != nil {
		if !real {
			t.Errorf("dbus.SystemBus() failed with fake Interface")
		}
		t.Skipf("D-Bus session bus is not available: %v", err)
	}
	busObj = bus.BusObject()

	name := fmt.Sprintf("io.kubernetes.dbus_test_%d", os.Getpid())
	owner := ""
	err = busObj.Call("org.freedesktop.DBus.GetNameOwner", 0, name).Store(&owner)
	if err == nil {
		t.Errorf("expected '%s' to be un-owned, but found owner %s", name, owner)
	}
	dbuserr, ok := err.(godbus.Error)
	if !ok {
		t.Errorf("expected godbus.Error, but got %#v", err)
	}
	if dbuserr.Name != "org.freedesktop.DBus.Error.NameHasNoOwner" {
		t.Errorf("expected NameHasNoOwner error but got %v", err)
	}

	sigchan := make(chan *godbus.Signal, 10)
	bus.Signal(sigchan)

	rule := fmt.Sprintf("type='signal',interface='org.freedesktop.DBus',member='NameOwnerChanged',path='/org/freedesktop/DBus',sender='org.freedesktop.DBus',arg0='%s'", name)
	err = busObj.Call("org.freedesktop.DBus.AddMatch", 0, rule).Store()
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	var ret uint32
	err = busObj.Call("org.freedesktop.DBus.RequestName", 0, name, DBusNameFlagDoNotQueue).Store(&ret)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if ret != DBusRequestNameReplyPrimaryOwner {
		t.Errorf("expected %v, got %v", DBusRequestNameReplyPrimaryOwner, ret)
	}

	err = busObj.Call("org.freedesktop.DBus.GetNameOwner", 0, name).Store(&owner)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	var changedSignal, acquiredSignal, lostSignal *godbus.Signal

	sig1 := <-sigchan
	sig2 := <-sigchan
	// We get two signals, but the order isn't guaranteed
	if sig1.Name == "org.freedesktop.DBus.NameOwnerChanged" {
		changedSignal = sig1
		acquiredSignal = sig2
	} else {
		acquiredSignal = sig1
		changedSignal = sig2
	}

	if acquiredSignal.Sender != "org.freedesktop.DBus" || acquiredSignal.Name != "org.freedesktop.DBus.NameAcquired" {
		t.Errorf("expected NameAcquired signal, got %v", acquiredSignal)
	}
	acquiredName := acquiredSignal.Body[0].(string)
	if acquiredName != name {
		t.Errorf("unexpected NameAcquired arguments: %v", acquiredSignal)
	}

	if changedSignal.Sender != "org.freedesktop.DBus" || changedSignal.Name != "org.freedesktop.DBus.NameOwnerChanged" {
		t.Errorf("expected NameOwnerChanged signal, got %v", changedSignal)
	}

	changedName := changedSignal.Body[0].(string)
	oldOwner := changedSignal.Body[1].(string)
	newOwner := changedSignal.Body[2].(string)
	if changedName != name || oldOwner != "" || newOwner != owner {
		t.Errorf("unexpected NameOwnerChanged arguments: %v", changedSignal)
	}

	err = busObj.Call("org.freedesktop.DBus.ReleaseName", 0, name).Store(&ret)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if ret != DBusReleaseNameReplyReleased {
		t.Errorf("expected %v, got %v", DBusReleaseNameReplyReleased, ret)
	}

	sig1 = <-sigchan
	sig2 = <-sigchan
	if sig1.Name == "org.freedesktop.DBus.NameOwnerChanged" {
		changedSignal = sig1
		lostSignal = sig2
	} else {
		lostSignal = sig1
		changedSignal = sig2
	}

	if lostSignal.Sender != "org.freedesktop.DBus" || lostSignal.Name != "org.freedesktop.DBus.NameLost" {
		t.Errorf("expected NameLost signal, got %v", lostSignal)
	}
	lostName := lostSignal.Body[0].(string)
	if lostName != name {
		t.Errorf("unexpected NameLost arguments: %v", lostSignal)
	}

	if changedSignal.Sender != "org.freedesktop.DBus" || changedSignal.Name != "org.freedesktop.DBus.NameOwnerChanged" {
		t.Errorf("expected NameOwnerChanged signal, got %v", changedSignal)
	}

	changedName = changedSignal.Body[0].(string)
	oldOwner = changedSignal.Body[1].(string)
	newOwner = changedSignal.Body[2].(string)
	if changedName != name || oldOwner != owner || newOwner != "" {
		t.Errorf("unexpected NameOwnerChanged arguments: %v", changedSignal)
	}

	if len(sigchan) != 0 {
		t.Errorf("unexpected extra signals (%d)", len(sigchan))
	}

	// Unregister sigchan
	bus.Signal(sigchan)
}

func TestRealDBus(t *testing.T) {
	dbus := New()
	doDBusTest(t, dbus, true)
}

func TestFakeDBus(t *testing.T) {
	uniqueName := ":1.1"
	ownedName := ""

	fakeSystem := NewFakeConnection()
	fakeSystem.SetBusObject(
		func(method string, args ...interface{}) ([]interface{}, error) {
			if method == "org.freedesktop.DBus.GetId" {
				return []interface{}{"foo"}, nil
			}
			return nil, fmt.Errorf("unexpected method call '%s'", method)
		},
	)

	fakeSession := NewFakeConnection()
	fakeSession.SetBusObject(
		func(method string, args ...interface{}) ([]interface{}, error) {
			if method == "org.freedesktop.DBus.GetNameOwner" {
				checkName := args[0].(string)
				if checkName != ownedName {
					return nil, godbus.Error{Name: "org.freedesktop.DBus.Error.NameHasNoOwner", Body: nil}
				} else {
					return []interface{}{uniqueName}, nil
				}
			} else if method == "org.freedesktop.DBus.RequestName" {
				reqName := args[0].(string)
				_ = args[1].(uint32)
				if ownedName != "" {
					return []interface{}{DBusRequestNameReplyAlreadyOwner}, nil
				}
				ownedName = reqName
				fakeSession.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameAcquired", reqName)
				fakeSession.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", reqName, "", uniqueName)
				return []interface{}{DBusRequestNameReplyPrimaryOwner}, nil
			} else if method == "org.freedesktop.DBus.ReleaseName" {
				reqName := args[0].(string)
				if reqName != ownedName {
					return []interface{}{DBusReleaseNameReplyNotOwner}, nil
				}
				ownedName = ""
				fakeSession.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", reqName, uniqueName, "")
				fakeSession.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameLost", reqName)
				return []interface{}{DBusReleaseNameReplyReleased}, nil
			} else if method == "org.freedesktop.DBus.AddMatch" {
				return nil, nil
			} else {
				return nil, fmt.Errorf("unexpected method call '%s'", method)
			}
		},
	)

	dbus := NewFake(fakeSystem, fakeSession)
	doDBusTest(t, dbus, false)
}
