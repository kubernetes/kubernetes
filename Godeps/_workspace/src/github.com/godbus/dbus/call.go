package dbus

import (
	"errors"
	"strings"
)

// Call represents a pending or completed method call.
type Call struct {
	Destination string
	Path        ObjectPath
	Method      string
	Args        []interface{}

	// Strobes when the call is complete.
	Done chan *Call

	// After completion, the error status. If this is non-nil, it may be an
	// error message from the peer (with Error as its type) or some other error.
	Err error

	// Holds the response once the call is done.
	Body []interface{}
}

var errSignature = errors.New("dbus: mismatched signature")

// Store stores the body of the reply into the provided pointers. It returns
// an error if the signatures of the body and retvalues don't match, or if
// the error status is not nil.
func (c *Call) Store(retvalues ...interface{}) error {
	if c.Err != nil {
		return c.Err
	}

	return Store(c.Body, retvalues...)
}

// Object represents a remote object on which methods can be invoked.
type Object struct {
	conn *Conn
	dest string
	path ObjectPath
}

// Call calls a method with (*Object).Go and waits for its reply.
func (o *Object) Call(method string, flags Flags, args ...interface{}) *Call {
	return <-o.Go(method, flags, make(chan *Call, 1), args...).Done
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

// Go calls a method with the given arguments asynchronously. It returns a
// Call structure representing this method call. The passed channel will
// return the same value once the call is done. If ch is nil, a new channel
// will be allocated. Otherwise, ch has to be buffered or Go will panic.
//
// If the flags include FlagNoReplyExpected, ch is ignored and a Call structure
// is returned of which only the Err member is valid.
//
// If the method parameter contains a dot ('.'), the part before the last dot
// specifies the interface on which the method is called.
func (o *Object) Go(method string, flags Flags, ch chan *Call, args ...interface{}) *Call {
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
		call := &Call{
			Destination: o.dest,
			Path:        o.path,
			Method:      method,
			Args:        args,
			Done:        ch,
		}
		o.conn.callsLck.Lock()
		o.conn.calls[msg.serial] = call
		o.conn.callsLck.Unlock()
		o.conn.outLck.RLock()
		if o.conn.closed {
			call.Err = ErrClosed
			call.Done <- call
		} else {
			o.conn.out <- msg
		}
		o.conn.outLck.RUnlock()
		return call
	}
	o.conn.outLck.RLock()
	defer o.conn.outLck.RUnlock()
	if o.conn.closed {
		return &Call{Err: ErrClosed}
	}
	o.conn.out <- msg
	return &Call{Err: nil}
}

// Destination returns the destination that calls on o are sent to.
func (o *Object) Destination() string {
	return o.dest
}

// Path returns the path that calls on o are sent to.
func (o *Object) Path() ObjectPath {
	return o.path
}
