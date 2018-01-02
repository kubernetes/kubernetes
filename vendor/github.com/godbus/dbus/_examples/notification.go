package main

import "github.com/godbus/dbus"

func main() {
	conn, err := dbus.SessionBus()
	if err != nil {
		panic(err)
	}
	obj := conn.Object("org.freedesktop.Notifications", "/org/freedesktop/Notifications")
	call := obj.Call("org.freedesktop.Notifications.Notify", 0, "", uint32(0),
		"", "Test", "This is a test of the DBus bindings for go.", []string{},
		map[string]dbus.Variant{}, int32(5000))
	if call.Err != nil {
		panic(call.Err)
	}
}
