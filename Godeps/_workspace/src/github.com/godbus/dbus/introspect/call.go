package introspect

import (
	"encoding/xml"
	"github.com/godbus/dbus"
	"strings"
)

// Call calls org.freedesktop.Introspectable.Introspect on a remote object
// and returns the introspection data.
func Call(o dbus.BusObject) (*Node, error) {
	var xmldata string
	var node Node

	err := o.Call("org.freedesktop.DBus.Introspectable.Introspect", 0).Store(&xmldata)
	if err != nil {
		return nil, err
	}
	err = xml.NewDecoder(strings.NewReader(xmldata)).Decode(&node)
	if err != nil {
		return nil, err
	}
	if node.Name == "" {
		node.Name = string(o.Path())
	}
	return &node, nil
}
