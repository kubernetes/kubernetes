package introspect

import (
	"encoding/xml"
	"github.com/godbus/dbus"
	"reflect"
	"strings"
)

// Introspectable implements org.freedesktop.Introspectable.
//
// You can create it by converting the XML-formatted introspection data from a
// string to an Introspectable or call NewIntrospectable with a Node. Then,
// export it as org.freedesktop.Introspectable on you object.
type Introspectable string

// NewIntrospectable returns an Introspectable that returns the introspection
// data that corresponds to the given Node. If n.Interfaces doesn't contain the
// data for org.freedesktop.DBus.Introspectable, it is added automatically.
func NewIntrospectable(n *Node) Introspectable {
	found := false
	for _, v := range n.Interfaces {
		if v.Name == "org.freedesktop.DBus.Introspectable" {
			found = true
			break
		}
	}
	if !found {
		n.Interfaces = append(n.Interfaces, IntrospectData)
	}
	b, err := xml.Marshal(n)
	if err != nil {
		panic(err)
	}
	return Introspectable(strings.TrimSpace(IntrospectDeclarationString) + string(b))
}

// Introspect implements org.freedesktop.Introspectable.Introspect.
func (i Introspectable) Introspect() (string, *dbus.Error) {
	return string(i), nil
}

// Methods returns the description of the methods of v. This can be used to
// create a Node which can be passed to NewIntrospectable.
func Methods(v interface{}) []Method {
	t := reflect.TypeOf(v)
	ms := make([]Method, 0, t.NumMethod())
	for i := 0; i < t.NumMethod(); i++ {
		if t.Method(i).PkgPath != "" {
			continue
		}
		mt := t.Method(i).Type
		if mt.NumOut() == 0 ||
			mt.Out(mt.NumOut()-1) != reflect.TypeOf(&dbus.Error{}) {

			continue
		}
		var m Method
		m.Name = t.Method(i).Name
		m.Args = make([]Arg, 0, mt.NumIn()+mt.NumOut()-2)
		for j := 1; j < mt.NumIn(); j++ {
			if mt.In(j) != reflect.TypeOf((*dbus.Sender)(nil)).Elem() {
				arg := Arg{"", dbus.SignatureOfType(mt.In(j)).String(), "in"}
				m.Args = append(m.Args, arg)
			}
		}
		for j := 0; j < mt.NumOut()-1; j++ {
			arg := Arg{"", dbus.SignatureOfType(mt.Out(j)).String(), "out"}
			m.Args = append(m.Args, arg)
		}
		m.Annotations = make([]Annotation, 0)
		ms = append(ms, m)
	}
	return ms
}
