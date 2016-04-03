package dbus

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

var (
	errmsgInvalidArg = Error{
		"org.freedesktop.DBus.Error.InvalidArgs",
		[]interface{}{"Invalid type / number of args"},
	}
	errmsgNoObject = Error{
		"org.freedesktop.DBus.Error.NoSuchObject",
		[]interface{}{"No such object"},
	}
	errmsgUnknownMethod = Error{
		"org.freedesktop.DBus.Error.UnknownMethod",
		[]interface{}{"Unknown / invalid method"},
	}
)

// exportWithMapping represents an exported struct along with a method name
// mapping to allow for exporting lower-case methods, etc.
type exportWithMapping struct {
	export interface{}

	// Method name mapping; key -> struct method, value -> dbus method.
	mapping map[string]string

	// Whether or not this export is for the entire subtree
	includeSubtree bool
}

// Sender is a type which can be used in exported methods to receive the message
// sender.
type Sender string

func exportedMethod(export exportWithMapping, name string) reflect.Value {
	if export.export == nil {
		return reflect.Value{}
	}

	// If a mapping was included in the export, check the map to see if we
	// should be looking for a different method in the export.
	if export.mapping != nil {
		for key, value := range export.mapping {
			if value == name {
				name = key
				break
			}

			// Catch the case where a method is aliased but the client is calling
			// the original, e.g. the "Foo" method was exported mapped to
			// "foo," and dbus client called the original "Foo."
			if key == name {
				return reflect.Value{}
			}
		}
	}

	value := reflect.ValueOf(export.export)
	m := value.MethodByName(name)

	// Catch the case of attempting to call an unexported method
	method, ok := value.Type().MethodByName(name)

	if !m.IsValid() || !ok || method.PkgPath != "" {
		return reflect.Value{}
	}
	t := m.Type()
	if t.NumOut() == 0 ||
		t.Out(t.NumOut()-1) != reflect.TypeOf(&errmsgInvalidArg) {

		return reflect.Value{}
	}
	return m
}

// searchHandlers will look through all registered handlers looking for one
// to handle the given path. If a verbatim one isn't found, it will check for
// a subtree registration for the path as well.
func (conn *Conn) searchHandlers(path ObjectPath) (map[string]exportWithMapping, bool) {
	conn.handlersLck.RLock()
	defer conn.handlersLck.RUnlock()

	handlers, ok := conn.handlers[path]
	if ok {
		return handlers, ok
	}

	// If handlers weren't found for this exact path, look for a matching subtree
	// registration
	handlers = make(map[string]exportWithMapping)
	path = path[:strings.LastIndex(string(path), "/")]
	for len(path) > 0 {
		var subtreeHandlers map[string]exportWithMapping
		subtreeHandlers, ok = conn.handlers[path]
		if ok {
			for iface, handler := range subtreeHandlers {
				// Only include this handler if it registered for the subtree
				if handler.includeSubtree {
					handlers[iface] = handler
				}
			}

			break
		}

		path = path[:strings.LastIndex(string(path), "/")]
	}

	return handlers, ok
}

// handleCall handles the given method call (i.e. looks if it's one of the
// pre-implemented ones and searches for a corresponding handler if not).
func (conn *Conn) handleCall(msg *Message) {
	name := msg.Headers[FieldMember].value.(string)
	path := msg.Headers[FieldPath].value.(ObjectPath)
	ifaceName, hasIface := msg.Headers[FieldInterface].value.(string)
	sender, hasSender := msg.Headers[FieldSender].value.(string)
	serial := msg.serial
	if ifaceName == "org.freedesktop.DBus.Peer" {
		switch name {
		case "Ping":
			conn.sendReply(sender, serial)
		case "GetMachineId":
			conn.sendReply(sender, serial, conn.uuid)
		default:
			conn.sendError(errmsgUnknownMethod, sender, serial)
		}
		return
	}
	if len(name) == 0 {
		conn.sendError(errmsgUnknownMethod, sender, serial)
	}

	// Find the exported handler (if any) for this path
	handlers, ok := conn.searchHandlers(path)
	if !ok {
		conn.sendError(errmsgNoObject, sender, serial)
		return
	}

	var m reflect.Value
	if hasIface {
		iface := handlers[ifaceName]
		m = exportedMethod(iface, name)
	} else {
		for _, v := range handlers {
			m = exportedMethod(v, name)
			if m.IsValid() {
				break
			}
		}
	}

	if !m.IsValid() {
		conn.sendError(errmsgUnknownMethod, sender, serial)
		return
	}

	t := m.Type()
	vs := msg.Body
	pointers := make([]interface{}, t.NumIn())
	decode := make([]interface{}, 0, len(vs))
	for i := 0; i < t.NumIn(); i++ {
		tp := t.In(i)
		val := reflect.New(tp)
		pointers[i] = val.Interface()
		if tp == reflect.TypeOf((*Sender)(nil)).Elem() {
			val.Elem().SetString(sender)
		} else if tp == reflect.TypeOf((*Message)(nil)).Elem() {
			val.Elem().Set(reflect.ValueOf(*msg))
		} else {
			decode = append(decode, pointers[i])
		}
	}

	if len(decode) != len(vs) {
		conn.sendError(errmsgInvalidArg, sender, serial)
		return
	}

	if err := Store(vs, decode...); err != nil {
		conn.sendError(errmsgInvalidArg, sender, serial)
		return
	}

	// Extract parameters
	params := make([]reflect.Value, len(pointers))
	for i := 0; i < len(pointers); i++ {
		params[i] = reflect.ValueOf(pointers[i]).Elem()
	}

	// Call method
	ret := m.Call(params)
	if em := ret[t.NumOut()-1].Interface().(*Error); em != nil {
		conn.sendError(*em, sender, serial)
		return
	}

	if msg.Flags&FlagNoReplyExpected == 0 {
		reply := new(Message)
		reply.Type = TypeMethodReply
		reply.serial = conn.getSerial()
		reply.Headers = make(map[HeaderField]Variant)
		if hasSender {
			reply.Headers[FieldDestination] = msg.Headers[FieldSender]
		}
		reply.Headers[FieldReplySerial] = MakeVariant(msg.serial)
		reply.Body = make([]interface{}, len(ret)-1)
		for i := 0; i < len(ret)-1; i++ {
			reply.Body[i] = ret[i].Interface()
		}
		if len(ret) != 1 {
			reply.Headers[FieldSignature] = MakeVariant(SignatureOf(reply.Body...))
		}
		conn.outLck.RLock()
		if !conn.closed {
			conn.out <- reply
		}
		conn.outLck.RUnlock()
	}
}

// Emit emits the given signal on the message bus. The name parameter must be
// formatted as "interface.member", e.g., "org.freedesktop.DBus.NameLost".
func (conn *Conn) Emit(path ObjectPath, name string, values ...interface{}) error {
	if !path.IsValid() {
		return errors.New("dbus: invalid object path")
	}
	i := strings.LastIndex(name, ".")
	if i == -1 {
		return errors.New("dbus: invalid method name")
	}
	iface := name[:i]
	member := name[i+1:]
	if !isValidMember(member) {
		return errors.New("dbus: invalid method name")
	}
	if !isValidInterface(iface) {
		return errors.New("dbus: invalid interface name")
	}
	msg := new(Message)
	msg.Type = TypeSignal
	msg.serial = conn.getSerial()
	msg.Headers = make(map[HeaderField]Variant)
	msg.Headers[FieldInterface] = MakeVariant(iface)
	msg.Headers[FieldMember] = MakeVariant(member)
	msg.Headers[FieldPath] = MakeVariant(path)
	msg.Body = values
	if len(values) > 0 {
		msg.Headers[FieldSignature] = MakeVariant(SignatureOf(values...))
	}
	conn.outLck.RLock()
	defer conn.outLck.RUnlock()
	if conn.closed {
		return ErrClosed
	}
	conn.out <- msg
	return nil
}

// Export registers the given value to be exported as an object on the
// message bus.
//
// If a method call on the given path and interface is received, an exported
// method with the same name is called with v as the receiver if the
// parameters match and the last return value is of type *Error. If this
// *Error is not nil, it is sent back to the caller as an error.
// Otherwise, a method reply is sent with the other return values as its body.
//
// Any parameters with the special type Sender are set to the sender of the
// dbus message when the method is called. Parameters of this type do not
// contribute to the dbus signature of the method (i.e. the method is exposed
// as if the parameters of type Sender were not there).
//
// Similarly, any parameters with the type Message are set to the raw message
// received on the bus. Again, parameters of this type do not contribute to the
// dbus signature of the method.
//
// Every method call is executed in a new goroutine, so the method may be called
// in multiple goroutines at once.
//
// Method calls on the interface org.freedesktop.DBus.Peer will be automatically
// handled for every object.
//
// Passing nil as the first parameter will cause conn to cease handling calls on
// the given combination of path and interface.
//
// Export returns an error if path is not a valid path name.
func (conn *Conn) Export(v interface{}, path ObjectPath, iface string) error {
	return conn.ExportWithMap(v, nil, path, iface)
}

// ExportWithMap works exactly like Export but provides the ability to remap
// method names (e.g. export a lower-case method).
//
// The keys in the map are the real method names (exported on the struct), and
// the values are the method names to be exported on DBus.
func (conn *Conn) ExportWithMap(v interface{}, mapping map[string]string, path ObjectPath, iface string) error {
	return conn.exportWithMap(v, mapping, path, iface, false)
}

// ExportSubtree works exactly like Export but registers the given value for
// an entire subtree rather under the root path provided.
//
// In order to make this useful, one parameter in each of the value's exported
// methods should be a Message, in which case it will contain the raw message
// (allowing one to get access to the path that caused the method to be called).
//
// Note that more specific export paths take precedence over less specific. For
// example, a method call using the ObjectPath /foo/bar/baz will call a method
// exported on /foo/bar before a method exported on /foo.
func (conn *Conn) ExportSubtree(v interface{}, path ObjectPath, iface string) error {
	return conn.ExportSubtreeWithMap(v, nil, path, iface)
}

// ExportSubtreeWithMap works exactly like ExportSubtree but provides the
// ability to remap method names (e.g. export a lower-case method).
//
// The keys in the map are the real method names (exported on the struct), and
// the values are the method names to be exported on DBus.
func (conn *Conn) ExportSubtreeWithMap(v interface{}, mapping map[string]string, path ObjectPath, iface string) error {
	return conn.exportWithMap(v, mapping, path, iface, true)
}

// exportWithMap is the worker function for all exports/registrations.
func (conn *Conn) exportWithMap(v interface{}, mapping map[string]string, path ObjectPath, iface string, includeSubtree bool) error {
	if !path.IsValid() {
		return fmt.Errorf(`dbus: Invalid path name: "%s"`, path)
	}

	conn.handlersLck.Lock()
	defer conn.handlersLck.Unlock()

	// Remove a previous export if the interface is nil
	if v == nil {
		if _, ok := conn.handlers[path]; ok {
			delete(conn.handlers[path], iface)
			if len(conn.handlers[path]) == 0 {
				delete(conn.handlers, path)
			}
		}

		return nil
	}

	// If this is the first handler for this path, make a new map to hold all
	// handlers for this path.
	if _, ok := conn.handlers[path]; !ok {
		conn.handlers[path] = make(map[string]exportWithMapping)
	}

	// Finally, save this handler
	conn.handlers[path][iface] = exportWithMapping{export: v, mapping: mapping, includeSubtree: includeSubtree}

	return nil
}

// ReleaseName calls org.freedesktop.DBus.ReleaseName and awaits a response.
func (conn *Conn) ReleaseName(name string) (ReleaseNameReply, error) {
	var r uint32
	err := conn.busObj.Call("org.freedesktop.DBus.ReleaseName", 0, name).Store(&r)
	if err != nil {
		return 0, err
	}
	return ReleaseNameReply(r), nil
}

// RequestName calls org.freedesktop.DBus.RequestName and awaits a response.
func (conn *Conn) RequestName(name string, flags RequestNameFlags) (RequestNameReply, error) {
	var r uint32
	err := conn.busObj.Call("org.freedesktop.DBus.RequestName", 0, name, flags).Store(&r)
	if err != nil {
		return 0, err
	}
	return RequestNameReply(r), nil
}

// ReleaseNameReply is the reply to a ReleaseName call.
type ReleaseNameReply uint32

const (
	ReleaseNameReplyReleased ReleaseNameReply = 1 + iota
	ReleaseNameReplyNonExistent
	ReleaseNameReplyNotOwner
)

// RequestNameFlags represents the possible flags for a RequestName call.
type RequestNameFlags uint32

const (
	NameFlagAllowReplacement RequestNameFlags = 1 << iota
	NameFlagReplaceExisting
	NameFlagDoNotQueue
)

// RequestNameReply is the reply to a RequestName call.
type RequestNameReply uint32

const (
	RequestNameReplyPrimaryOwner RequestNameReply = 1 + iota
	RequestNameReplyInQueue
	RequestNameReplyExists
	RequestNameReplyAlreadyOwner
)
