package systemd

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	dbus "github.com/godbus/dbus/v5"
	"golang.org/x/sys/unix"
)

var (
	dbusC        *systemdDbus.Conn
	dbusMu       sync.RWMutex
	dbusInited   bool
	dbusRootless bool
)

type dbusConnManager struct{}

// newDbusConnManager initializes systemd dbus connection manager.
func newDbusConnManager(rootless bool) *dbusConnManager {
	dbusMu.Lock()
	defer dbusMu.Unlock()
	if dbusInited && rootless != dbusRootless {
		panic("can't have both root and rootless dbus")
	}
	dbusInited = true
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
		// When dbus-user-session is not installed, we can't detect whether we should try to connect to user dbus or system dbus, so d.dbusRootless is set to false.
		// This may fail with a cryptic error "read unix @->/run/systemd/private: read: connection reset by peer: unknown."
		// https://github.com/moby/moby/issues/42793
		return nil, fmt.Errorf("failed to connect to dbus (hint: for rootless containers, maybe you need to install dbus-user-session package, see https://github.com/opencontainers/runc/blob/master/docs/cgroup-v2.md): %w", err)
	}
	dbusC = conn
	return conn, nil
}

func (d *dbusConnManager) newConnection() (*systemdDbus.Conn, error) {
	newDbusConn := func() (*systemdDbus.Conn, error) {
		if dbusRootless {
			return newUserSystemdDbus()
		}
		return systemdDbus.NewWithContext(context.TODO())
	}

	var err error
	for retry := range 7 {
		var conn *systemdDbus.Conn
		conn, err = newDbusConn()
		if !errors.Is(err, unix.EAGAIN) {
			return conn, err
		}
		// Exponential backoff (100ms * 2^attempt + ~12.5% jitter).
		// At most we would expect 15 seconds of delay with 7 attempts.
		delay := 100 * time.Millisecond << retry
		delay += time.Duration(rand.Int64N(1 + (delay.Milliseconds() >> 3)))
		time.Sleep(delay)
	}
	return nil, fmt.Errorf("dbus connection failed after several retries: %w", err)
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
		if err == nil {
			return nil
		}
		if !errors.Is(err, dbus.ErrClosed) {
			return err
		}
		d.resetConnection(conn)
	}
}
