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

package dbus

import (
	"errors"
	"log"
	"time"

	"github.com/godbus/dbus/v5"
)

const (
	cleanIgnoreInterval = int64(10 * time.Second)
	ignoreInterval      = int64(30 * time.Millisecond)
)

// Subscribe sets up this connection to subscribe to all systemd dbus events.
// This is required before calling SubscribeUnits. When the connection closes
// systemd will automatically stop sending signals so there is no need to
// explicitly call Unsubscribe().
func (c *Conn) Subscribe() error {
	c.sigconn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
		"type='signal',interface='org.freedesktop.systemd1.Manager',member='UnitNew'")
	c.sigconn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0,
		"type='signal',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'")

	return c.sigobj.Call("org.freedesktop.systemd1.Manager.Subscribe", 0).Store()
}

// Unsubscribe this connection from systemd dbus events.
func (c *Conn) Unsubscribe() error {
	return c.sigobj.Call("org.freedesktop.systemd1.Manager.Unsubscribe", 0).Store()
}

func (c *Conn) dispatch() {
	ch := make(chan *dbus.Signal, signalBuffer)

	c.sigconn.Signal(ch)

	go func() {
		for {
			signal, ok := <-ch
			if !ok {
				return
			}

			if signal.Name == "org.freedesktop.systemd1.Manager.JobRemoved" {
				c.jobComplete(signal)
			}

			if c.subStateSubscriber.updateCh == nil &&
				c.propertiesSubscriber.updateCh == nil {
				continue
			}

			var unitPath dbus.ObjectPath
			switch signal.Name {
			case "org.freedesktop.systemd1.Manager.JobRemoved":
				unitName := signal.Body[2].(string)
				_ = c.sysobj.Call("org.freedesktop.systemd1.Manager.GetUnit", 0, unitName).Store(&unitPath)
			case "org.freedesktop.systemd1.Manager.UnitNew":
				unitPath = signal.Body[1].(dbus.ObjectPath)
			case "org.freedesktop.DBus.Properties.PropertiesChanged":
				if signal.Body[0].(string) == "org.freedesktop.systemd1.Unit" {
					unitPath = signal.Path

					if len(signal.Body) >= 2 {
						if changed, ok := signal.Body[1].(map[string]dbus.Variant); ok {
							c.sendPropertiesUpdate(unitPath, changed)
						}
					}
				}
			}

			if unitPath == dbus.ObjectPath("") {
				continue
			}

			c.sendSubStateUpdate(unitPath)
		}
	}()
}

// SubscribeUnits returns two unbuffered channels which will receive all changed units every
// interval.  Deleted units are sent as nil.
func (c *Conn) SubscribeUnits(interval time.Duration) (<-chan map[string]*UnitStatus, <-chan error) {
	return c.SubscribeUnitsCustom(interval, 0, func(u1, u2 *UnitStatus) bool { return *u1 != *u2 }, nil)
}

// SubscribeUnitsCustom is like SubscribeUnits but lets you specify the buffer
// size of the channels, the comparison function for detecting changes and a filter
// function for cutting down on the noise that your channel receives.
func (c *Conn) SubscribeUnitsCustom(interval time.Duration, buffer int, isChanged func(*UnitStatus, *UnitStatus) bool, filterUnit func(string) bool) (<-chan map[string]*UnitStatus, <-chan error) {
	old := make(map[string]*UnitStatus)
	statusChan := make(chan map[string]*UnitStatus, buffer)
	errChan := make(chan error, buffer)

	go func() {
		for {
			timerChan := time.After(interval)

			units, err := c.ListUnits()
			if err == nil {
				cur := make(map[string]*UnitStatus)
				for i := range units {
					if filterUnit != nil && filterUnit(units[i].Name) {
						continue
					}
					cur[units[i].Name] = &units[i]
				}

				// add all new or changed units
				changed := make(map[string]*UnitStatus)
				for n, u := range cur {
					if oldU, ok := old[n]; !ok || isChanged(oldU, u) {
						changed[n] = u
					}
					delete(old, n)
				}

				// add all deleted units
				for oldN := range old {
					changed[oldN] = nil
				}

				old = cur

				if len(changed) != 0 {
					statusChan <- changed
				}
			} else {
				errChan <- err
			}

			<-timerChan
		}
	}()

	return statusChan, errChan
}

type SubStateUpdate struct {
	UnitName string
	SubState string
}

// SetSubStateSubscriber writes to updateCh when any unit's substate changes.
// Although this writes to updateCh on every state change, the reported state
// may be more recent than the change that generated it (due to an unavoidable
// race in the systemd dbus interface).  That is, this method provides a good
// way to keep a current view of all units' states, but is not guaranteed to
// show every state transition they go through.  Furthermore, state changes
// will only be written to the channel with non-blocking writes.  If updateCh
// is full, it attempts to write an error to errCh; if errCh is full, the error
// passes silently.
func (c *Conn) SetSubStateSubscriber(updateCh chan<- *SubStateUpdate, errCh chan<- error) {
	if c == nil {
		msg := "nil receiver"
		select {
		case errCh <- errors.New(msg):
		default:
			log.Printf("full error channel while reporting: %s\n", msg)
		}
		return
	}

	c.subStateSubscriber.Lock()
	defer c.subStateSubscriber.Unlock()
	c.subStateSubscriber.updateCh = updateCh
	c.subStateSubscriber.errCh = errCh
}

func (c *Conn) sendSubStateUpdate(unitPath dbus.ObjectPath) {
	c.subStateSubscriber.Lock()
	defer c.subStateSubscriber.Unlock()

	if c.subStateSubscriber.updateCh == nil {
		return
	}

	isIgnored := c.shouldIgnore(unitPath)
	defer c.cleanIgnore()
	if isIgnored {
		return
	}

	info, err := c.GetUnitPathProperties(unitPath)
	if err != nil {
		select {
		case c.subStateSubscriber.errCh <- err:
		default:
			log.Printf("full error channel while reporting: %s\n", err)
		}
		return
	}
	defer c.updateIgnore(unitPath, info)

	name, ok := info["Id"].(string)
	if !ok {
		msg := "failed to cast info.Id"
		select {
		case c.subStateSubscriber.errCh <- errors.New(msg):
		default:
			log.Printf("full error channel while reporting: %s\n", err)
		}
		return
	}
	substate, ok := info["SubState"].(string)
	if !ok {
		msg := "failed to cast info.SubState"
		select {
		case c.subStateSubscriber.errCh <- errors.New(msg):
		default:
			log.Printf("full error channel while reporting: %s\n", msg)
		}
		return
	}

	update := &SubStateUpdate{name, substate}
	select {
	case c.subStateSubscriber.updateCh <- update:
	default:
		msg := "update channel is full"
		select {
		case c.subStateSubscriber.errCh <- errors.New(msg):
		default:
			log.Printf("full error channel while reporting: %s\n", msg)
		}
		return
	}
}

// The ignore functions work around a wart in the systemd dbus interface.
// Requesting the properties of an unloaded unit will cause systemd to send a
// pair of UnitNew/UnitRemoved signals.  Because we need to get a unit's
// properties on UnitNew (as that's the only indication of a new unit coming up
// for the first time), we would enter an infinite loop if we did not attempt
// to detect and ignore these spurious signals.  The signal themselves are
// indistinguishable from relevant ones, so we (somewhat hackishly) ignore an
// unloaded unit's signals for a short time after requesting its properties.
// This means that we will miss e.g. a transient unit being restarted
// *immediately* upon failure and also a transient unit being started
// immediately after requesting its status (with systemctl status, for example,
// because this causes a UnitNew signal to be sent which then causes us to fetch
// the properties).

func (c *Conn) shouldIgnore(path dbus.ObjectPath) bool {
	t, ok := c.subStateSubscriber.ignore[path]
	return ok && t >= time.Now().UnixNano()
}

func (c *Conn) updateIgnore(path dbus.ObjectPath, info map[string]any) {
	loadState, ok := info["LoadState"].(string)
	if !ok {
		return
	}

	// unit is unloaded - it will trigger bad systemd dbus behavior
	if loadState == "not-found" {
		c.subStateSubscriber.ignore[path] = time.Now().UnixNano() + ignoreInterval
	}
}

// without this, ignore would grow unboundedly over time
func (c *Conn) cleanIgnore() {
	now := time.Now().UnixNano()
	if c.subStateSubscriber.cleanIgnore < now {
		c.subStateSubscriber.cleanIgnore = now + cleanIgnoreInterval

		for p, t := range c.subStateSubscriber.ignore {
			if t < now {
				delete(c.subStateSubscriber.ignore, p)
			}
		}
	}
}

// PropertiesUpdate holds a map of a unit's changed properties
type PropertiesUpdate struct {
	UnitName string
	Changed  map[string]dbus.Variant
}

// SetPropertiesSubscriber writes to updateCh when any unit's properties
// change. Every property change reported by systemd will be sent; that is, no
// transitions will be "missed" (as they might be with SetSubStateSubscriber).
// However, state changes will only be written to the channel with non-blocking
// writes.  If updateCh is full, it attempts to write an error to errCh; if
// errCh is full, the error passes silently.
func (c *Conn) SetPropertiesSubscriber(updateCh chan<- *PropertiesUpdate, errCh chan<- error) {
	c.propertiesSubscriber.Lock()
	defer c.propertiesSubscriber.Unlock()
	c.propertiesSubscriber.updateCh = updateCh
	c.propertiesSubscriber.errCh = errCh
}

// we don't need to worry about shouldIgnore() here because
// sendPropertiesUpdate doesn't call GetProperties()
func (c *Conn) sendPropertiesUpdate(unitPath dbus.ObjectPath, changedProps map[string]dbus.Variant) {
	c.propertiesSubscriber.Lock()
	defer c.propertiesSubscriber.Unlock()

	if c.propertiesSubscriber.updateCh == nil {
		return
	}

	update := &PropertiesUpdate{unitName(unitPath), changedProps}

	select {
	case c.propertiesSubscriber.updateCh <- update:
	default:
		msg := "update channel is full"
		select {
		case c.propertiesSubscriber.errCh <- errors.New(msg):
		default:
			log.Printf("full error channel while reporting: %s\n", msg)
		}
		return
	}
}
