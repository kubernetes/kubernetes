package dbus

import "os"

const defaultSystemBusAddress = "tcp:host=127.0.0.1,port=12434"

func getSystemBusPlatformAddress() (string, error) {
	address := os.Getenv("DBUS_SYSTEM_BUS_ADDRESS")
	if address != "" {
		return address, nil
	}
	return defaultSystemBusAddress, nil
}
