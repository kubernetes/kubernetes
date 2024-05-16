package dbus

import (
	"context"
	"errors"
	"strings"
)

// BusObject is the interface of a remote object on which methods can be
// invoked.
type BusObject interface {
	Call(method string, flags Flags, args ...interface{}) *Call
	CallWithContext(ctx context.Context, method string, flags Flags, args ...interface{}) *Call
	Go(method string, flags Flags, ch chan *Call, args ...interface{}) *Call
	GoWithContext(ctx context.Context, method string, flags Flags, ch chan *Call, args ...interface{}) *Call
	AddMatchSignal(iface, member string, options ...MatchOption) *Call
	RemoveMatchSignal(iface, member string, options ...MatchOption) *Call
	GetProperty(p string) (Variant, error)
	StoreProperty(p string, value interface{}) error
	SetProperty(p string, v interface{}) error
	Destination() string
	Path() ObjectPath
}

// Object represents a remote object on which methods can be invoked.
type Object struct {
	conn *Conn
	dest string
	path ObjectPath
}

// Call calls a method with (*Object).Go and waits for its reply.
func (o *Object) Call(method string, flags Flags, args ...interface{}) *Call {
	return <-o.createCall(context.Background(), method, flags, make(chan *Call, 1), args...).Done
}

// CallWithContext acts like Call but takes a context
func (o *Object) CallWithContext(ctx context.Context, method string, flags Flags, args ...interface{}) *Call {
	return <-o.createCall(ctx, method, flags, make(chan *Call, 1), args...).Done
}

// AddMatchSignal subscribes BusObject to signals from specified interface,
// method (member). Additional filter rules can be added via WithMatch* option constructors.
// Note: To filter events by object path you have to specify this path via an option.
//
// Deprecated: use (*Conn) AddMatchSignal instead.
func (o *Object) AddMatchSignal(iface, member string, options ...MatchOption) *Call {
	base := []MatchOption{
		withMatchType("signal"),
		WithMatchInterface(iface),
		WithMatchMember(member),
	}

	options = append(base, options...)
	return o.conn.BusObject().Call(
		"org.freedesktop.DBus.AddMatch",
		0,
		formatMatchOptions(options),
	)
}

// RemoveMatchSignal unsubscribes BusObject from signals from specified interface,
// method (member). Additional filter rules can be added via WithMatch* option constructors
//
// Deprecated: use (*Conn) RemoveMatchSignal instead.
func (o *Object) RemoveMatchSignal(iface, member string, options ...MatchOption) *Call {
	base := []MatchOption{
		withMatchType("signal"),
		WithMatchInterface(iface),
		WithMatchMember(member),
	}

	options = append(base, options...)
	return o.conn.BusObject().Call(
		"org.freedesktop.DBus.RemoveMatch",
		0,
		formatMatchOptions(options),
	)
}

// Go calls a method with the given arguments asynchronously. It returns a
// Call structure representing this method call. The passed channel will
// return the same value once the call is done. If ch is nil, a new channel
// will be allocated. Otherwise, ch has to be buffered or Go will panic.
//
// If the flags include FlagNoReplyExpected, ch is ignored and a Call structure
// is returned with any error in Err and a closed channel in Done containing
// the returned Call as it's one entry.
//
// If the method parameter contains a dot ('.'), the part before the last dot
// specifies the interface on which the method is called.
func (o *Object) Go(method string, flags Flags, ch chan *Call, args ...interface{}) *Call {
	return o.createCall(context.Background(), method, flags, ch, args...)
}

// GoWithContext acts like Go but takes a context
func (o *Object) GoWithContext(ctx context.Context, method string, flags Flags, ch chan *Call, args ...interface{}) *Call {
	return o.createCall(ctx, method, flags, ch, args...)
}

func (o *Object) createCall(ctx context.Context, method string, flags Flags, ch chan *Call, args ...interface{}) *Call {
	if ctx == nil {
		panic("nil context")
	}
	iface := ""
	i := strings.LastIndex(method, ".")
	if i != -1 {
		iface = method[:i]
	}
	method = method[i+1:]
	msg := new(Message)
	msg.Type = TypeMethodCall
	msg.Flags = flags & (FlagNoAutoStart | FlagNoReplyExpected)
	msg.Headers = make(map[HeaderField]Variant)
	msg.Headers[FieldPath] = MakeVariant(o.path)
	msg.Headers[FieldDestination] = MakeVariant(o.dest)
	msg.Headers[FieldMember] = MakeVariant(method)
	if iface != "" {
		msg.Headers[FieldInterface] = MakeVariant(iface)
	}
	msg.Body = args
	if len(args) > 0 {
		msg.Headers[FieldSignature] = MakeVariant(SignatureOf(args...))
	}
	return o.conn.SendWithContext(ctx, msg, ch)
}

// GetProperty calls org.freedesktop.DBus.Properties.Get on the given
// object. The property name must be given in interface.member notation.
func (o *Object) GetProperty(p string) (Variant, error) {
	var result Variant
	err := o.StoreProperty(p, &result)
	return result, err
}

// StoreProperty calls org.freedesktop.DBus.Properties.Get on the given
// object. The property name must be given in interface.member notation.
// It stores the returned property into the provided value.
func (o *Object) StoreProperty(p string, value interface{}) error {
	idx := strings.LastIndex(p, ".")
	if idx == -1 || idx+1 == len(p) {
		return errors.New("dbus: invalid property " + p)
	}

	iface := p[:idx]
	prop := p[idx+1:]

	return o.Call("org.freedesktop.DBus.Properties.Get", 0, iface, prop).
		Store(value)
}

// SetProperty calls org.freedesktop.DBus.Properties.Set on the given
// object. The property name must be given in interface.member notation.
func (o *Object) SetProperty(p string, v interface{}) error {
	idx := strings.LastIndex(p, ".")
	if idx == -1 || idx+1 == len(p) {
		return errors.New("dbus: invalid property " + p)
	}

	iface := p[:idx]
	prop := p[idx+1:]

	return o.Call("org.freedesktop.DBus.Properties.Set", 0, iface, prop, v).Err
}

// Destination returns the destination that calls on (o *Object) are sent to.
func (o *Object) Destination() string {
	return o.dest
}

// Path returns the path that calls on (o *Object") are sent to.
func (o *Object) Path() ObjectPath {
	return o.path
}
