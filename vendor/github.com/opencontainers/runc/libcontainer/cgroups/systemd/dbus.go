// +build linux

package systemd

import (
	"context"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	dbus "github.com/godbus/dbus/v5"
)

var (
	dbusC        *systemdDbus.Conn
	dbusMu       sync.RWMutex
	dbusInited   bool
	dbusRootless bool
)

type dbusConnManager struct {
}

// newDbusConnManager initializes systemd dbus connection manager.
func newDbusConnManager(rootless bool) *dbusConnManager {
	if dbusInited && rootless != dbusRootless {
		panic("can't have both root and rootless dbus")
	}
	dbusRootless = rootless
	return &dbusConnManager{}
}

// getConnection lazily initializes and returns systemd dbus connection.
func (d *dbusConnManager) getConnection() (*systemdDbus.Conn, error) {
	// In the case where dbusC != nil
	// Use the read lock the first time to ensure
	// that Conn can be acquired at the same time.
	dbusMu.RLock()
	if conn := dbusC; conn != nil {
		dbusMu.RUnlock()
		return conn, nil
	}
	dbusMu.RUnlock()

	// In the case where dbusC == nil
	// Use write lock to ensure that only one
	// will be created
	dbusMu.Lock()
	defer dbusMu.Unlock()
	if conn := dbusC; conn != nil {
		return conn, nil
	}

	conn, err := d.newConnection()
	if err != nil {
		return nil, err
	}
	dbusC = conn
	return conn, nil
}

func (d *dbusConnManager) newConnection() (*systemdDbus.Conn, error) {
	if dbusRootless {
		return newUserSystemdDbus()
	}
	return systemdDbus.NewWithContext(context.TODO())
}

// resetConnection resets the connection to its initial state
// (so it can be reconnected if necessary).
func (d *dbusConnManager) resetConnection(conn *systemdDbus.Conn) {
	dbusMu.Lock()
	defer dbusMu.Unlock()
	if dbusC != nil && dbusC == conn {
		dbusC.Close()
		dbusC = nil
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
