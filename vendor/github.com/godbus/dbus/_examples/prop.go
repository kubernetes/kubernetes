package main

import (
	"fmt"
	"github.com/godbus/dbus"
	"github.com/godbus/dbus/introspect"
	"github.com/godbus/dbus/prop"
	"os"
)

type foo string

func (f foo) Foo() (string, *dbus.Error) {
	fmt.Println(f)
	return string(f), nil
}

func main() {
	conn, err := dbus.SessionBus()
	if err != nil {
		panic(err)
	}
	reply, err := conn.RequestName("com.github.guelfey.Demo",
		dbus.NameFlagDoNotQueue)
	if err != nil {
		panic(err)
	}
	if reply != dbus.RequestNameReplyPrimaryOwner {
		fmt.Fprintln(os.Stderr, "name already taken")
		os.Exit(1)
	}
	propsSpec := map[string]map[string]*prop.Prop{
		"com.github.guelfey.Demo": {
			"SomeInt": {
				int32(0),
				true,
				prop.EmitTrue,
				func(c *prop.Change) *dbus.Error {
					fmt.Println(c.Name, "changed to", c.Value)
					return nil
				},
			},
		},
	}
	f := foo("Bar")
	conn.Export(f, "/com/github/guelfey/Demo", "com.github.guelfey.Demo")
	props := prop.New(conn, "/com/github/guelfey/Demo", propsSpec)
	n := &introspect.Node{
		Name: "/com/github/guelfey/Demo",
		Interfaces: []introspect.Interface{
			introspect.IntrospectData,
			prop.IntrospectData,
			{
				Name:       "com.github.guelfey.Demo",
				Methods:    introspect.Methods(f),
				Properties: props.Introspection("com.github.guelfey.Demo"),
			},
		},
	}
	conn.Export(introspect.NewIntrospectable(n), "/com/github/guelfey/Demo",
		"org.freedesktop.DBus.Introspectable")
	fmt.Println("Listening on com.github.guelfey.Demo / /com/github/guelfey/Demo ...")

	c := make(chan *dbus.Signal)
	conn.Signal(c)
	for _ = range c {
	}
}
