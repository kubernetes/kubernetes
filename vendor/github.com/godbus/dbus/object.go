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

// MatchOption specifies option for dbus routing match rule. Options can be constructed with WithMatch* helpers.
// For full list of available options consult
// https://dbus.freedesktop.org/doc/dbus-specification.html#message-bus-routing-match-rules
type MatchOption struct {
	key   string
	value string
}

// WithMatchOption creates match option with given key and value
func WithMatchOption(key, value string) MatchOption {
	return MatchOption{key, value}
}

// WithMatchObjectPath creates match option that filters events based on given path
func WithMatchObjectPath(path ObjectPath) MatchOption {
	return MatchOption{"path", string(path)}
}

func formatMatchOptions(options []MatchOption) string {
	items := make([]string, 0, len(options))
	for _, option := range options {
		items = append(items, option.key+"='"+option.value+"'")
	}

	return strings.Join(items, ",")
}

// AddMatchSignal subscribes BusObject to signals from specified interface,
// method (member). Additional filter rules can be added via WithMatch* option constructors.
// Note: To filter events by object path you have to specify this path via an option.
func (o *Object) AddMatchSignal(iface, member string, options ...MatchOption) *Call {
	base := []MatchOption{
		{"type", "signal"},
		{"interface", iface},
		{"member", member},
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
func (o *Object) RemoveMatchSignal(iface, member string, options ...MatchOption) *Call {
	base := []MatchOption{
		{"type", "signal"},
		{"interface", iface},
		{"member", member},
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
	msg.serial = o.conn.getSerial()
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
	if msg.Flags&FlagNoReplyExpected == 0 {
		if ch == nil {
			ch = make(chan *Call, 10)
		} else if cap(ch) == 0 {
			panic("dbus: unbuffered channel passed to (*Object).Go")
		}
		ctx, cancel := context.WithCancel(ctx)
		call := &Call{
			Destination: o.dest,
			Path:        o.path,
			Method:      method,
			Args:        args,
			Done:        ch,
			ctxCanceler: cancel,
			ctx:         ctx,
		}
		o.conn.calls.track(msg.serial, call)
		o.conn.sendMessageAndIfClosed(msg, func() {
			o.conn.calls.handleSendError(msg, ErrClosed)
			cancel()
		})
		go func() {
			<-ctx.Done()
			o.conn.calls.handleSendError(msg, ctx.Err())
		}()

		return call
	}
	done := make(chan *Call, 1)
	call := &Call{
		Err:  nil,
		Done: done,
	}
	defer func() {
		call.Done <- call
		close(done)
	}()
	o.conn.sendMessageAndIfClosed(msg, func() {
		call.Err = ErrClosed
	})
	return call
}

// GetProperty calls org.freedesktop.DBus.Properties.GetProperty on the given
// object. The property name must be given in interface.member notation.
func (o *Object) GetProperty(p string) (Variant, error) {
	idx := strings.LastIndex(p, ".")
	if idx == -1 || idx+1 == len(p) {
		return Variant{}, errors.New("dbus: invalid property " + p)
	}

	iface := p[:idx]
	prop := p[idx+1:]

	result := Variant{}
	err := o.Call("org.freedesktop.DBus.Properties.Get", 0, iface, prop).Store(&result)

	if err != nil {
		return Variant{}, err
	}

	return result, nil
}

// Destination returns the destination that calls on (o *Object) are sent to.
func (o *Object) Destination() string {
	return o.dest
}

// Path returns the path that calls on (o *Object") are sent to.
func (o *Object) Path() ObjectPath {
	return o.path
}
