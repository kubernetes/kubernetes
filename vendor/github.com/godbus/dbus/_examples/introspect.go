package main

import (
	"encoding/json"
	"github.com/godbus/dbus"
	"github.com/godbus/dbus/introspect"
	"os"
)

func main() {
	conn, err := dbus.SessionBus()
	if err != nil {
		panic(err)
	}
	node, err := introspect.Call(conn.Object("org.freedesktop.DBus", "/org/freedesktop/DBus"))
	if err != nil {
		panic(err)
	}
	data, _ := json.MarshalIndent(node, "", "    ")
	os.Stdout.Write(data)
}
