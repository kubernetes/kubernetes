package dbus

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

var (
	ErrMsgInvalidArg = Error{
		"org.freedesktop.DBus.Error.InvalidArgs",
		[]interface{}{"Invalid type / number of args"},
	}
	ErrMsgNoObject = Error{
		"org.freedesktop.DBus.Error.NoSuchObject",
		[]interface{}{"No such object"},
	}
	ErrMsgUnknownMethod = Error{
		"org.freedesktop.DBus.Error.UnknownMethod",
		[]interface{}{"Unknown / invalid method"},
	}
	ErrMsgUnknownInterface = Error{
		"org.freedesktop.DBus.Error.UnknownInterface",
		[]interface{}{"Object does not implement the interface"},
	}
)

func MakeFailedError(err error) *Error {
	return &Error{
		"org.freedesktop.DBus.Error.Failed",
		[]interface{}{err.Error()},
	}
}

// Sender is a type which can be used in exported methods to receive the message
// sender.
type Sender string

func computeMethodName(name string, mapping map[string]string) string {
	newname, ok := mapping[name]
	if ok {
		name = newname
	}
	return name
}

func getMethods(in interface{}, mapping map[string]string) map[string]reflect.Value {
	if in == nil {
		return nil
	}
	methods := make(map[string]reflect.Value)
	val := reflect.ValueOf(in)
	typ := val.Type()
	for i := 0; i < typ.NumMethod(); i++ {
		methtype := typ.Method(i)
		method := val.Method(i)
		t := method.Type()
		// only track valid methods must return *Error as last arg
		// and must be exported
		if t.NumOut() == 0 ||
			t.Out(t.NumOut()-1) != reflect.TypeOf(&ErrMsgInvalidArg) ||
			methtype.PkgPath != "" {
			continue
		}
		// map names while building table
		methods[computeMethodName(methtype.Name, mapping)] = method
	}
	return methods
}

func standardMethodArgumentDecode(m Method, sender string, msg *Message, body []interface{}) ([]interface{}, error) {
	pointers := make([]interface{}, m.NumArguments())
	decode := make([]interface{}, 0, len(body))

	for i := 0; i < m.NumArguments(); i++ {
		tp := reflect.TypeOf(m.ArgumentValue(i))
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

	if len(decode) != len(body) {
		return nil, ErrMsgInvalidArg
	}

	if err := Store(body, decode...); err != nil {
		return nil, ErrMsgInvalidArg
	}

	return pointers, nil
}

func (conn *Conn) decodeArguments(m Method, sender string, msg *Message) ([]interface{}, error) {
	if decoder, ok := m.(ArgumentDecoder); ok {
		return decoder.DecodeArguments(conn, sender, msg, msg.Body)
	}
	return standardMethodArgumentDecode(m, sender, msg, msg.Body)
}

// handleCall handles the given method call (i.e. looks if it's one of the
// pre-implemented ones and searches for a corresponding handler if not).
func (conn *Conn) handleCall(msg *Message) {
	name := msg.Headers[FieldMember].value.(string)
	path := msg.Headers[FieldPath].value.(ObjectPath)
	ifaceName, _ := msg.Headers[FieldInterface].value.(string)
	sender, hasSender := msg.Headers[FieldSender].value.(string)
	serial := msg.serial
	if ifaceName == "org.freedesktop.DBus.Peer" {
		switch name {
		case "Ping":
			conn.sendReply(sender, serial)
		case "GetMachineId":
			conn.sendReply(sender, serial, conn.uuid)
		default:
			conn.sendError(ErrMsgUnknownMethod, sender, serial)
		}
		return
	}
	if len(name) == 0 {
		conn.sendError(ErrMsgUnknownMethod, sender, serial)
	}

	object, ok := conn.handler.LookupObject(path)
	if !ok {
		conn.sendError(ErrMsgNoObject, sender, serial)
		return
	}

	iface, exists := object.LookupInterface(ifaceName)
	if !exists {
		conn.sendError(ErrMsgUnknownInterface, sender, serial)
		return
	}

	m, exists := iface.LookupMethod(name)
	if !exists {
		conn.sendError(ErrMsgUnknownMethod, sender, serial)
		return
	}
	args, err := conn.decodeArguments(m, sender, msg)
	if err != nil {
		conn.sendError(err, sender, serial)
		return
	}

	ret, err := m.Call(args...)
	if err != nil {
		conn.sendError(err, sender, serial)
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
		reply.Body = make([]interface{}, len(ret))
		for i := 0; i < len(ret); i++ {
			reply.Body[i] = ret[i]
		}
		reply.Headers[FieldSignature] = MakeVariant(SignatureOf(reply.Body...))

		conn.sendMessage(reply)
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

	var closed bool
	conn.sendMessageAndIfClosed(msg, func() {
		closed = true
	})
	if closed {
		return ErrClosed
	}
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
	return conn.export(getMethods(v, mapping), path, iface, false)
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
	return conn.export(getMethods(v, mapping), path, iface, true)
}

// ExportMethodTable like Export registers the given methods as an object
// on the message bus. Unlike Export the it uses a method table to define
// the object instead of a native go object.
//
// The method table is a map from method name to function closure
// representing the method. This allows an object exported on the bus to not
// necessarily be a native go object. It can be useful for generating exposed
// methods on the fly.
//
// Any non-function objects in the method table are ignored.
func (conn *Conn) ExportMethodTable(methods map[string]interface{}, path ObjectPath, iface string) error {
	return conn.exportMethodTable(methods, path, iface, false)
}

// Like ExportSubtree, but with the same caveats as ExportMethodTable.
func (conn *Conn) ExportSubtreeMethodTable(methods map[string]interface{}, path ObjectPath, iface string) error {
	return conn.exportMethodTable(methods, path, iface, true)
}

func (conn *Conn) exportMethodTable(methods map[string]interface{}, path ObjectPath, iface string, includeSubtree bool) error {
	out := make(map[string]reflect.Value)
	for name, method := range methods {
		rval := reflect.ValueOf(method)
		if rval.Kind() != reflect.Func {
			continue
		}
		t := rval.Type()
		// only track valid methods must return *Error as last arg
		if t.NumOut() == 0 ||
			t.Out(t.NumOut()-1) != reflect.TypeOf(&ErrMsgInvalidArg) {
			continue
		}
		out[name] = rval
	}
	return conn.export(out, path, iface, includeSubtree)
}

func (conn *Conn) unexport(h *defaultHandler, path ObjectPath, iface string) error {
	if h.PathExists(path) {
		obj := h.objects[path]
		obj.DeleteInterface(iface)
		if len(obj.interfaces) == 0 {
			h.DeleteObject(path)
		}
	}
	return nil
}

// exportWithMap is the worker function for all exports/registrations.
func (conn *Conn) export(methods map[string]reflect.Value, path ObjectPath, iface string, includeSubtree bool) error {
	h, ok := conn.handler.(*defaultHandler)
	if !ok {
		return fmt.Errorf(
			`dbus: export only allowed on the default hander handler have %T"`,
			conn.handler)
	}

	if !path.IsValid() {
		return fmt.Errorf(`dbus: Invalid path name: "%s"`, path)
	}

	// Remove a previous export if the interface is nil
	if methods == nil {
		return conn.unexport(h, path, iface)
	}

	// If this is the first handler for this path, make a new map to hold all
	// handlers for this path.
	if !h.PathExists(path) {
		h.AddObject(path, newExportedObject())
	}

	exportedMethods := make(map[string]Method)
	for name, method := range methods {
		exportedMethods[name] = exportedMethod{method}
	}

	// Finally, save this handler
	obj := h.objects[path]
	obj.AddInterface(iface, newExportedIntf(exportedMethods, includeSubtree))

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
