//go:build !windows && !solaris && !darwin
// +build !windows,!solaris,!darwin

package dbus

import (
	"net"
	"os"
)

const defaultSystemBusAddress = "unix:path=/var/run/dbus/system_bus_socket"

func getSystemBusPlatformAddress() string {
	address := os.Getenv("DBUS_SYSTEM_BUS_ADDRESS")
	if address != "" {
		return address
	}
	return defaultSystemBusAddress
}

// DialUnix establishes a new private connection to the message bus specified by UnixConn.
func DialUnix(conn *net.UnixConn, opts ...ConnOption) (*Conn, error) {
	tr := newUnixTransportFromConn(conn)
	return newConn(tr, opts...)
}

func ConnectUnix(uconn *net.UnixConn, opts ...ConnOption) (*Conn, error) {
	conn, err := DialUnix(uconn, opts...)
	if err != nil {
		return nil, err
	}
	if err = conn.Auth(conn.auth); err != nil {
		_ = conn.Close()
		return nil, err
	}
	if err = conn.Hello(); err != nil {
		_ = conn.Close()
		return nil, err
	}
	return conn, nil
}
