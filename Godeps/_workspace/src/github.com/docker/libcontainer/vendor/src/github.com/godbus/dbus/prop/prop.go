// Package prop provides the Properties struct which can be used to implement
// org.freedesktop.DBus.Properties.
package prop

import (
	"github.com/godbus/dbus"
	"github.com/godbus/dbus/introspect"
	"sync"
)

// EmitType controls how org.freedesktop.DBus.Properties.PropertiesChanged is
// emitted for a property. If it is EmitTrue, the signal is emitted. If it is
// EmitInvalidates, the signal is also emitted, but the new value of the property
// is not disclosed.
type EmitType byte

const (
	EmitFalse EmitType = iota
	EmitTrue
	EmitInvalidates
)

// ErrIfaceNotFound is the error returned to peers who try to access properties
// on interfaces that aren't found.
var ErrIfaceNotFound = &dbus.Error{"org.freedesktop.DBus.Properties.Error.InterfaceNotFound", nil}

// ErrPropNotFound is the error returned to peers trying to access properties
// that aren't found.
var ErrPropNotFound = &dbus.Error{"org.freedesktop.DBus.Properties.Error.PropertyNotFound", nil}

// ErrReadOnly is the error returned to peers trying to set a read-only
// property.
var ErrReadOnly = &dbus.Error{"org.freedesktop.DBus.Properties.Error.ReadOnly", nil}

// ErrInvalidArg is returned to peers if the type of the property that is being
// changed and the argument don't match.
var ErrInvalidArg = &dbus.Error{"org.freedesktop.DBus.Properties.Error.InvalidArg", nil}

// The introspection data for the org.freedesktop.DBus.Properties interface.
var IntrospectData = introspect.Interface{
	Name: "org.freedesktop.DBus.Properties",
	Methods: []introspect.Method{
		{
			Name: "Get",
			Args: []introspect.Arg{
				{"interface", "s", "in"},
				{"property", "s", "in"},
				{"value", "v", "out"},
			},
		},
		{
			Name: "GetAll",
			Args: []introspect.Arg{
				{"interface", "s", "in"},
				{"props", "a{sv}", "out"},
			},
		},
		{
			Name: "Set",
			Args: []introspect.Arg{
				{"interface", "s", "in"},
				{"property", "s", "in"},
				{"value", "v", "in"},
			},
		},
	},
	Signals: []introspect.Signal{
		{
			Name: "PropertiesChanged",
			Args: []introspect.Arg{
				{"interface", "s", "out"},
				{"changed_properties", "a{sv}", "out"},
				{"invalidates_properties", "as", "out"},
			},
		},
	},
}

// The introspection data for the org.freedesktop.DBus.Properties interface, as
// a string.
const IntrospectDataString = `
	<interface name="org.freedesktop.DBus.Introspectable">
		<method name="Get">
			<arg name="interface" direction="in" type="s"/>
			<arg name="property" direction="in" type="s"/>
			<arg name="value" direction="out" type="v"/>
		</method>
		<method name="GetAll">
			<arg name="interface" direction="in" type="s"/>
			<arg name="props" direction="out" type="a{sv}"/>
		</method>
		<method name="Set">
			<arg name="interface" direction="in" type="s"/>
			<arg name="property" direction="in" type="s"/>
			<arg name="value" direction="in" type="v"/>
		</method>
		<signal name="PropertiesChanged">
			<arg name="interface" type="s"/>
			<arg name="changed_properties" type="a{sv}"/>
			<arg name="invalidates_properties" type="as"/>
		</signal>
	</interface>
`

// Prop represents a single property. It is used for creating a Properties
// value.
type Prop struct {
	// Initial value. Must be a DBus-representable type.
	Value interface{}

	// If true, the value can be modified by calls to Set.
	Writable bool

	// Controls how org.freedesktop.DBus.Properties.PropertiesChanged is
	// emitted if this property changes.
	Emit EmitType

	// If not nil, anytime this property is changed by Set, this function is
	// called with an appropiate Change as its argument. If the returned error
	// is not nil, it is sent back to the caller of Set and the property is not
	// changed.
	Callback func(*Change) *dbus.Error
}

// Change represents a change of a property by a call to Set.
type Change struct {
	Props *Properties
	Iface string
	Name  string
	Value interface{}
}

// Properties is a set of values that can be made available to the message bus
// using the org.freedesktop.DBus.Properties interface. It is safe for
// concurrent use by multiple goroutines.
type Properties struct {
	m    map[string]map[string]*Prop
	mut  sync.RWMutex
	conn *dbus.Conn
	path dbus.ObjectPath
}

// New returns a new Properties structure that manages the given properties.
// The key for the first-level map of props is the name of the interface; the
// second-level key is the name of the property. The returned structure will be
// exported as org.freedesktop.DBus.Properties on path.
func New(conn *dbus.Conn, path dbus.ObjectPath, props map[string]map[string]*Prop) *Properties {
	p := &Properties{m: props, conn: conn, path: path}
	conn.Export(p, path, "org.freedesktop.DBus.Properties")
	return p
}

// Get implements org.freedesktop.DBus.Properties.Get.
func (p *Properties) Get(iface, property string) (dbus.Variant, *dbus.Error) {
	p.mut.RLock()
	defer p.mut.RUnlock()
	m, ok := p.m[iface]
	if !ok {
		return dbus.Variant{}, ErrIfaceNotFound
	}
	prop, ok := m[property]
	if !ok {
		return dbus.Variant{}, ErrPropNotFound
	}
	return dbus.MakeVariant(prop.Value), nil
}

// GetAll implements org.freedesktop.DBus.Properties.GetAll.
func (p *Properties) GetAll(iface string) (map[string]dbus.Variant, *dbus.Error) {
	p.mut.RLock()
	defer p.mut.RUnlock()
	m, ok := p.m[iface]
	if !ok {
		return nil, ErrIfaceNotFound
	}
	rm := make(map[string]dbus.Variant, len(m))
	for k, v := range m {
		rm[k] = dbus.MakeVariant(v.Value)
	}
	return rm, nil
}

// GetMust returns the value of the given property and panics if either the
// interface or the property name are invalid.
func (p *Properties) GetMust(iface, property string) interface{} {
	p.mut.RLock()
	defer p.mut.RUnlock()
	return p.m[iface][property].Value
}

// Introspection returns the introspection data that represents the properties
// of iface.
func (p *Properties) Introspection(iface string) []introspect.Property {
	p.mut.RLock()
	defer p.mut.RUnlock()
	m := p.m[iface]
	s := make([]introspect.Property, 0, len(m))
	for k, v := range m {
		p := introspect.Property{Name: k, Type: dbus.SignatureOf(v.Value).String()}
		if v.Writable {
			p.Access = "readwrite"
		} else {
			p.Access = "read"
		}
		s = append(s, p)
	}
	return s
}

// set sets the given property and emits PropertyChanged if appropiate. p.mut
// must already be locked.
func (p *Properties) set(iface, property string, v interface{}) {
	prop := p.m[iface][property]
	prop.Value = v
	switch prop.Emit {
	case EmitFalse:
		// do nothing
	case EmitInvalidates:
		p.conn.Emit(p.path, "org.freedesktop.DBus.Properties.PropertiesChanged",
			iface, map[string]dbus.Variant{}, []string{property})
	case EmitTrue:
		p.conn.Emit(p.path, "org.freedesktop.DBus.Properties.PropertiesChanged",
			iface, map[string]dbus.Variant{property: dbus.MakeVariant(v)},
			[]string{})
	default:
		panic("invalid value for EmitType")
	}
}

// Set implements org.freedesktop.Properties.Set.
func (p *Properties) Set(iface, property string, newv dbus.Variant) *dbus.Error {
	p.mut.Lock()
	defer p.mut.Unlock()
	m, ok := p.m[iface]
	if !ok {
		return ErrIfaceNotFound
	}
	prop, ok := m[property]
	if !ok {
		return ErrPropNotFound
	}
	if !prop.Writable {
		return ErrReadOnly
	}
	if newv.Signature() != dbus.SignatureOf(prop.Value) {
		return ErrInvalidArg
	}
	if prop.Callback != nil {
		err := prop.Callback(&Change{p, iface, property, newv.Value()})
		if err != nil {
			return err
		}
	}
	p.set(iface, property, newv.Value())
	return nil
}

// SetMust sets the value of the given property and panics if the interface or
// the property name are invalid.
func (p *Properties) SetMust(iface, property string, v interface{}) {
	p.mut.Lock()
	p.set(iface, property, v)
	p.mut.Unlock()
}
