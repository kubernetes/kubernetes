// +build linux

package systemd

import (
	"context"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	dbus "github.com/godbus/dbus/v5"
)

type dbusConnManager struct {
	conn     *systemdDbus.Conn
	rootless bool
	sync.RWMutex
}

// newDbusConnManager initializes systemd dbus connection manager.
func newDbusConnManager(rootless bool) *dbusConnManager {
	return &dbusConnManager{
		rootless: rootless,
	}
}

// getConnection lazily initializes and returns systemd dbus connection.
func (d *dbusConnManager) getConnection() (*systemdDbus.Conn, error) {
	// In the case where d.conn != nil
	// Use the read lock the first time to ensure
	// that Conn can be acquired at the same time.
	d.RLock()
	if conn := d.conn; conn != nil {
		d.RUnlock()
		return conn, nil
	}
	d.RUnlock()

	// In the case where d.conn == nil
	// Use write lock to ensure that only one
	// will be created
	d.Lock()
	defer d.Unlock()
	if conn := d.conn; conn != nil {
		return conn, nil
	}

	conn, err := d.newConnection()
	if err != nil {
		return nil, err
	}
	d.conn = conn
	return conn, nil
}

func (d *dbusConnManager) newConnection() (*systemdDbus.Conn, error) {
	if d.rootless {
		return newUserSystemdDbus()
	}
	return systemdDbus.NewWithContext(context.TODO())
}

// resetConnection resets the connection to its initial state
// (so it can be reconnected if necessary).
func (d *dbusConnManager) resetConnection(conn *systemdDbus.Conn) {
	d.Lock()
	defer d.Unlock()
	if d.conn != nil && d.conn == conn {
		d.conn.Close()
		d.conn = nil
	}
}

var errDbusConnClosed = dbus.ErrClosed.Error()

// retryOnDisconnect calls op, and if the error it returns is about closed dbus
// connection, the connection is re-established and the op is retried. This helps
// with the situation when dbus is restarted and we have a stale connection.
func (d *dbusConnManager) retryOnDisconnect(op func(*systemdDbus.Conn) error) error {
	for {
		conn, err := d.getConnection()
		if err != nil {
			return err
		}
		err = op(conn)
		if !isDbusError(err, errDbusConnClosed) {
			return err
		}
		d.resetConnection(conn)
	}
}
