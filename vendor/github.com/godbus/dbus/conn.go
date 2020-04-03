package dbus

import (
	"context"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"sync"
)

var (
	systemBus     *Conn
	systemBusLck  sync.Mutex
	sessionBus    *Conn
	sessionBusLck sync.Mutex
)

// ErrClosed is the error returned by calls on a closed connection.
var ErrClosed = errors.New("dbus: connection closed by user")

// Conn represents a connection to a message bus (usually, the system or
// session bus).
//
// Connections are either shared or private. Shared connections
// are shared between calls to the functions that return them. As a result,
// the methods Close, Auth and Hello must not be called on them.
//
// Multiple goroutines may invoke methods on a connection simultaneously.
type Conn struct {
	transport

	busObj BusObject
	unixFD bool
	uuid   string

	handler       Handler
	signalHandler SignalHandler
	serialGen     SerialGenerator

	names      *nameTracker
	calls      *callTracker
	outHandler *outputHandler

	eavesdropped    chan<- *Message
	eavesdroppedLck sync.Mutex
}

// SessionBus returns a shared connection to the session bus, connecting to it
// if not already done.
func SessionBus() (conn *Conn, err error) {
	sessionBusLck.Lock()
	defer sessionBusLck.Unlock()
	if sessionBus != nil {
		return sessionBus, nil
	}
	defer func() {
		if conn != nil {
			sessionBus = conn
		}
	}()
	conn, err = SessionBusPrivate()
	if err != nil {
		return
	}
	if err = conn.Auth(nil); err != nil {
		conn.Close()
		conn = nil
		return
	}
	if err = conn.Hello(); err != nil {
		conn.Close()
		conn = nil
	}
	return
}

func getSessionBusAddress() (string, error) {
	if address := os.Getenv("DBUS_SESSION_BUS_ADDRESS"); address != "" && address != "autolaunch:" {
		return address, nil

	} else if address := tryDiscoverDbusSessionBusAddress(); address != "" {
		os.Setenv("DBUS_SESSION_BUS_ADDRESS", address)
		return address, nil
	}
	return getSessionBusPlatformAddress()
}

// SessionBusPrivate returns a new private connection to the session bus.
func SessionBusPrivate(opts ...ConnOption) (*Conn, error) {
	address, err := getSessionBusAddress()
	if err != nil {
		return nil, err
	}

	return Dial(address, opts...)
}

// SessionBusPrivate returns a new private connection to the session bus.
//
// Deprecated: use SessionBusPrivate with options instead.
func SessionBusPrivateHandler(handler Handler, signalHandler SignalHandler) (*Conn, error) {
	return SessionBusPrivate(WithHandler(handler), WithSignalHandler(signalHandler))
}

// SystemBus returns a shared connection to the system bus, connecting to it if
// not already done.
func SystemBus() (conn *Conn, err error) {
	systemBusLck.Lock()
	defer systemBusLck.Unlock()
	if systemBus != nil {
		return systemBus, nil
	}
	defer func() {
		if conn != nil {
			systemBus = conn
		}
	}()
	conn, err = SystemBusPrivate()
	if err != nil {
		return
	}
	if err = conn.Auth(nil); err != nil {
		conn.Close()
		conn = nil
		return
	}
	if err = conn.Hello(); err != nil {
		conn.Close()
		conn = nil
	}
	return
}

// SystemBusPrivate returns a new private connection to the system bus.
// Note: this connection is not ready to use. One must perform Auth and Hello
// on the connection before it is useable.
func SystemBusPrivate(opts ...ConnOption) (*Conn, error) {
	return Dial(getSystemBusPlatformAddress(), opts...)
}

// SystemBusPrivateHandler returns a new private connection to the system bus, using the provided handlers.
//
// Deprecated: use SystemBusPrivate with options instead.
func SystemBusPrivateHandler(handler Handler, signalHandler SignalHandler) (*Conn, error) {
	return SystemBusPrivate(WithHandler(handler), WithSignalHandler(signalHandler))
}

// Dial establishes a new private connection to the message bus specified by address.
func Dial(address string, opts ...ConnOption) (*Conn, error) {
	tr, err := getTransport(address)
	if err != nil {
		return nil, err
	}
	return newConn(tr, opts...)
}

// DialHandler establishes a new private connection to the message bus specified by address, using the supplied handlers.
//
// Deprecated: use Dial with options instead.
func DialHandler(address string, handler Handler, signalHandler SignalHandler) (*Conn, error) {
	return Dial(address, WithSignalHandler(signalHandler))
}

// ConnOption is a connection option.
type ConnOption func(conn *Conn) error

// WithHandler overrides the default handler.
func WithHandler(handler Handler) ConnOption {
	return func(conn *Conn) error {
		conn.handler = handler
		return nil
	}
}

// WithSignalHandler overrides the default signal handler.
func WithSignalHandler(handler SignalHandler) ConnOption {
	return func(conn *Conn) error {
		conn.signalHandler = handler
		return nil
	}
}

// WithSerialGenerator overrides the default signals generator.
func WithSerialGenerator(gen SerialGenerator) ConnOption {
	return func(conn *Conn) error {
		conn.serialGen = gen
		return nil
	}
}

// NewConn creates a new private *Conn from an already established connection.
func NewConn(conn io.ReadWriteCloser, opts ...ConnOption) (*Conn, error) {
	return newConn(genericTransport{conn}, opts...)
}

// NewConnHandler creates a new private *Conn from an already established connection, using the supplied handlers.
//
// Deprecated: use NewConn with options instead.
func NewConnHandler(conn io.ReadWriteCloser, handler Handler, signalHandler SignalHandler) (*Conn, error) {
	return NewConn(genericTransport{conn}, WithHandler(handler), WithSignalHandler(signalHandler))
}

// newConn creates a new *Conn from a transport.
func newConn(tr transport, opts ...ConnOption) (*Conn, error) {
	conn := new(Conn)
	conn.transport = tr
	for _, opt := range opts {
		if err := opt(conn); err != nil {
			return nil, err
		}
	}
	conn.calls = newCallTracker()
	if conn.handler == nil {
		conn.handler = NewDefaultHandler()
	}
	if conn.signalHandler == nil {
		conn.signalHandler = NewDefaultSignalHandler()
	}
	if conn.serialGen == nil {
		conn.serialGen = newSerialGenerator()
	}
	conn.outHandler = &outputHandler{conn: conn}
	conn.names = newNameTracker()
	conn.busObj = conn.Object("org.freedesktop.DBus", "/org/freedesktop/DBus")
	return conn, nil
}

// BusObject returns the object owned by the bus daemon which handles
// administrative requests.
func (conn *Conn) BusObject() BusObject {
	return conn.busObj
}

// Close closes the connection. Any blocked operations will return with errors
// and the channels passed to Eavesdrop and Signal are closed. This method must
// not be called on shared connections.
func (conn *Conn) Close() error {
	conn.outHandler.close()
	if term, ok := conn.signalHandler.(Terminator); ok {
		term.Terminate()
	}

	if term, ok := conn.handler.(Terminator); ok {
		term.Terminate()
	}

	conn.eavesdroppedLck.Lock()
	if conn.eavesdropped != nil {
		close(conn.eavesdropped)
	}
	conn.eavesdroppedLck.Unlock()

	return conn.transport.Close()
}

// Eavesdrop causes conn to send all incoming messages to the given channel
// without further processing. Method replies, errors and signals will not be
// sent to the appropiate channels and method calls will not be handled. If nil
// is passed, the normal behaviour is restored.
//
// The caller has to make sure that ch is sufficiently buffered;
// if a message arrives when a write to ch is not possible, the message is
// discarded.
func (conn *Conn) Eavesdrop(ch chan<- *Message) {
	conn.eavesdroppedLck.Lock()
	conn.eavesdropped = ch
	conn.eavesdroppedLck.Unlock()
}

// getSerial returns an unused serial.
func (conn *Conn) getSerial() uint32 {
	return conn.serialGen.GetSerial()
}

// Hello sends the initial org.freedesktop.DBus.Hello call. This method must be
// called after authentication, but before sending any other messages to the
// bus. Hello must not be called for shared connections.
func (conn *Conn) Hello() error {
	var s string
	err := conn.busObj.Call("org.freedesktop.DBus.Hello", 0).Store(&s)
	if err != nil {
		return err
	}
	conn.names.acquireUniqueConnectionName(s)
	return nil
}

// inWorker runs in an own goroutine, reading incoming messages from the
// transport and dispatching them appropiately.
func (conn *Conn) inWorker() {
	for {
		msg, err := conn.ReadMessage()
		if err != nil {
			if _, ok := err.(InvalidMessageError); !ok {
				// Some read error occured (usually EOF); we can't really do
				// anything but to shut down all stuff and returns errors to all
				// pending replies.
				conn.Close()
				conn.calls.finalizeAllWithError(err)
				return
			}
			// invalid messages are ignored
			continue
		}
		conn.eavesdroppedLck.Lock()
		if conn.eavesdropped != nil {
			select {
			case conn.eavesdropped <- msg:
			default:
			}
			conn.eavesdroppedLck.Unlock()
			continue
		}
		conn.eavesdroppedLck.Unlock()
		dest, _ := msg.Headers[FieldDestination].value.(string)
		found := dest == "" ||
			!conn.names.uniqueNameIsKnown() ||
			conn.names.isKnownName(dest)
		if !found {
			// Eavesdropped a message, but no channel for it is registered.
			// Ignore it.
			continue
		}
		switch msg.Type {
		case TypeError:
			conn.serialGen.RetireSerial(conn.calls.handleDBusError(msg))
		case TypeMethodReply:
			conn.serialGen.RetireSerial(conn.calls.handleReply(msg))
		case TypeSignal:
			conn.handleSignal(msg)
		case TypeMethodCall:
			go conn.handleCall(msg)
		}

	}
}

func (conn *Conn) handleSignal(msg *Message) {
	iface := msg.Headers[FieldInterface].value.(string)
	member := msg.Headers[FieldMember].value.(string)
	// as per http://dbus.freedesktop.org/doc/dbus-specification.html ,
	// sender is optional for signals.
	sender, _ := msg.Headers[FieldSender].value.(string)
	if iface == "org.freedesktop.DBus" && sender == "org.freedesktop.DBus" {
		if member == "NameLost" {
			// If we lost the name on the bus, remove it from our
			// tracking list.
			name, ok := msg.Body[0].(string)
			if !ok {
				panic("Unable to read the lost name")
			}
			conn.names.loseName(name)
		} else if member == "NameAcquired" {
			// If we acquired the name on the bus, add it to our
			// tracking list.
			name, ok := msg.Body[0].(string)
			if !ok {
				panic("Unable to read the acquired name")
			}
			conn.names.acquireName(name)
		}
	}
	signal := &Signal{
		Sender: sender,
		Path:   msg.Headers[FieldPath].value.(ObjectPath),
		Name:   iface + "." + member,
		Body:   msg.Body,
	}
	conn.signalHandler.DeliverSignal(iface, member, signal)
}

// Names returns the list of all names that are currently owned by this
// connection. The slice is always at least one element long, the first element
// being the unique name of the connection.
func (conn *Conn) Names() []string {
	return conn.names.listKnownNames()
}

// Object returns the object identified by the given destination name and path.
func (conn *Conn) Object(dest string, path ObjectPath) BusObject {
	return &Object{conn, dest, path}
}

func (conn *Conn) sendMessage(msg *Message) {
	conn.sendMessageAndIfClosed(msg, func() {})
}

func (conn *Conn) sendMessageAndIfClosed(msg *Message, ifClosed func()) {
	err := conn.outHandler.sendAndIfClosed(msg, ifClosed)
	conn.calls.handleSendError(msg, err)
	if err != nil {
		conn.serialGen.RetireSerial(msg.serial)
	} else if msg.Type != TypeMethodCall {
		conn.serialGen.RetireSerial(msg.serial)
	}
}

// Send sends the given message to the message bus. You usually don't need to
// use this; use the higher-level equivalents (Call / Go, Emit and Export)
// instead. If msg is a method call and NoReplyExpected is not set, a non-nil
// call is returned and the same value is sent to ch (which must be buffered)
// once the call is complete. Otherwise, ch is ignored and a Call structure is
// returned of which only the Err member is valid.
func (conn *Conn) Send(msg *Message, ch chan *Call) *Call {
	return conn.send(context.Background(), msg, ch)
}

// SendWithContext acts like Send but takes a context
func (conn *Conn) SendWithContext(ctx context.Context, msg *Message, ch chan *Call) *Call {
	return conn.send(ctx, msg, ch)
}

func (conn *Conn) send(ctx context.Context, msg *Message, ch chan *Call) *Call {
	if ctx == nil {
		panic("nil context")
	}

	var call *Call
	ctx, canceler := context.WithCancel(ctx)
	msg.serial = conn.getSerial()
	if msg.Type == TypeMethodCall && msg.Flags&FlagNoReplyExpected == 0 {
		if ch == nil {
			ch = make(chan *Call, 5)
		} else if cap(ch) == 0 {
			panic("dbus: unbuffered channel passed to (*Conn).Send")
		}
		call = new(Call)
		call.Destination, _ = msg.Headers[FieldDestination].value.(string)
		call.Path, _ = msg.Headers[FieldPath].value.(ObjectPath)
		iface, _ := msg.Headers[FieldInterface].value.(string)
		member, _ := msg.Headers[FieldMember].value.(string)
		call.Method = iface + "." + member
		call.Args = msg.Body
		call.Done = ch
		call.ctx = ctx
		call.ctxCanceler = canceler
		conn.calls.track(msg.serial, call)
		go func() {
			<-ctx.Done()
			conn.calls.handleSendError(msg, ctx.Err())
		}()
		conn.sendMessageAndIfClosed(msg, func() {
			conn.calls.handleSendError(msg, ErrClosed)
			canceler()
		})
	} else {
		canceler()
		call = &Call{Err: nil}
		conn.sendMessageAndIfClosed(msg, func() {
			call = &Call{Err: ErrClosed}
		})
	}
	return call
}

// sendError creates an error message corresponding to the parameters and sends
// it to conn.out.
func (conn *Conn) sendError(err error, dest string, serial uint32) {
	var e *Error
	switch em := err.(type) {
	case Error:
		e = &em
	case *Error:
		e = em
	case DBusError:
		name, body := em.DBusError()
		e = NewError(name, body)
	default:
		e = MakeFailedError(err)
	}
	msg := new(Message)
	msg.Type = TypeError
	msg.serial = conn.getSerial()
	msg.Headers = make(map[HeaderField]Variant)
	if dest != "" {
		msg.Headers[FieldDestination] = MakeVariant(dest)
	}
	msg.Headers[FieldErrorName] = MakeVariant(e.Name)
	msg.Headers[FieldReplySerial] = MakeVariant(serial)
	msg.Body = e.Body
	if len(e.Body) > 0 {
		msg.Headers[FieldSignature] = MakeVariant(SignatureOf(e.Body...))
	}
	conn.sendMessage(msg)
}

// sendReply creates a method reply message corresponding to the parameters and
// sends it to conn.out.
func (conn *Conn) sendReply(dest string, serial uint32, values ...interface{}) {
	msg := new(Message)
	msg.Type = TypeMethodReply
	msg.serial = conn.getSerial()
	msg.Headers = make(map[HeaderField]Variant)
	if dest != "" {
		msg.Headers[FieldDestination] = MakeVariant(dest)
	}
	msg.Headers[FieldReplySerial] = MakeVariant(serial)
	msg.Body = values
	if len(values) > 0 {
		msg.Headers[FieldSignature] = MakeVariant(SignatureOf(values...))
	}
	conn.sendMessage(msg)
}

func (conn *Conn) defaultSignalAction(fn func(h *defaultSignalHandler, ch chan<- *Signal), ch chan<- *Signal) {
	if !isDefaultSignalHandler(conn.signalHandler) {
		return
	}
	handler := conn.signalHandler.(*defaultSignalHandler)
	fn(handler, ch)
}

// Signal registers the given channel to be passed all received signal messages.
// The caller has to make sure that ch is sufficiently buffered; if a message
// arrives when a write to c is not possible, it is discarded.
//
// Multiple of these channels can be registered at the same time.
//
// These channels are "overwritten" by Eavesdrop; i.e., if there currently is a
// channel for eavesdropped messages, this channel receives all signals, and
// none of the channels passed to Signal will receive any signals.
func (conn *Conn) Signal(ch chan<- *Signal) {
	conn.defaultSignalAction((*defaultSignalHandler).addSignal, ch)
}

// RemoveSignal removes the given channel from the list of the registered channels.
func (conn *Conn) RemoveSignal(ch chan<- *Signal) {
	conn.defaultSignalAction((*defaultSignalHandler).removeSignal, ch)
}

// SupportsUnixFDs returns whether the underlying transport supports passing of
// unix file descriptors. If this is false, method calls containing unix file
// descriptors will return an error and emitted signals containing them will
// not be sent.
func (conn *Conn) SupportsUnixFDs() bool {
	return conn.unixFD
}

// Error represents a D-Bus message of type Error.
type Error struct {
	Name string
	Body []interface{}
}

func NewError(name string, body []interface{}) *Error {
	return &Error{name, body}
}

func (e Error) Error() string {
	if len(e.Body) >= 1 {
		s, ok := e.Body[0].(string)
		if ok {
			return s
		}
	}
	return e.Name
}

// Signal represents a D-Bus message of type Signal. The name member is given in
// "interface.member" notation, e.g. org.freedesktop.D-Bus.NameLost.
type Signal struct {
	Sender string
	Path   ObjectPath
	Name   string
	Body   []interface{}
}

// transport is a D-Bus transport.
type transport interface {
	// Read and Write raw data (for example, for the authentication protocol).
	io.ReadWriteCloser

	// Send the initial null byte used for the EXTERNAL mechanism.
	SendNullByte() error

	// Returns whether this transport supports passing Unix FDs.
	SupportsUnixFDs() bool

	// Signal the transport that Unix FD passing is enabled for this connection.
	EnableUnixFDs()

	// Read / send a message, handling things like Unix FDs.
	ReadMessage() (*Message, error)
	SendMessage(*Message) error
}

var (
	transports = make(map[string]func(string) (transport, error))
)

func getTransport(address string) (transport, error) {
	var err error
	var t transport

	addresses := strings.Split(address, ";")
	for _, v := range addresses {
		i := strings.IndexRune(v, ':')
		if i == -1 {
			err = errors.New("dbus: invalid bus address (no transport)")
			continue
		}
		f := transports[v[:i]]
		if f == nil {
			err = errors.New("dbus: invalid bus address (invalid or unsupported transport)")
			continue
		}
		t, err = f(v[i+1:])
		if err == nil {
			return t, nil
		}
	}
	return nil, err
}

// dereferenceAll returns a slice that, assuming that vs is a slice of pointers
// of arbitrary types, containes the values that are obtained from dereferencing
// all elements in vs.
func dereferenceAll(vs []interface{}) []interface{} {
	for i := range vs {
		v := reflect.ValueOf(vs[i])
		v = v.Elem()
		vs[i] = v.Interface()
	}
	return vs
}

// getKey gets a key from a the list of keys. Returns "" on error / not found...
func getKey(s, key string) string {
	for _, keyEqualsValue := range strings.Split(s, ",") {
		keyValue := strings.SplitN(keyEqualsValue, "=", 2)
		if len(keyValue) == 2 && keyValue[0] == key {
			return keyValue[1]
		}
	}
	return ""
}

type outputHandler struct {
	conn    *Conn
	sendLck sync.Mutex
	closed  struct {
		isClosed bool
		lck      sync.RWMutex
	}
}

func (h *outputHandler) sendAndIfClosed(msg *Message, ifClosed func()) error {
	h.closed.lck.RLock()
	defer h.closed.lck.RUnlock()
	if h.closed.isClosed {
		ifClosed()
		return nil
	}
	h.sendLck.Lock()
	defer h.sendLck.Unlock()
	return h.conn.SendMessage(msg)
}

func (h *outputHandler) close() {
	h.closed.lck.Lock()
	defer h.closed.lck.Unlock()
	h.closed.isClosed = true
}

type serialGenerator struct {
	lck        sync.Mutex
	nextSerial uint32
	serialUsed map[uint32]bool
}

func newSerialGenerator() *serialGenerator {
	return &serialGenerator{
		serialUsed: map[uint32]bool{0: true},
		nextSerial: 1,
	}
}

func (gen *serialGenerator) GetSerial() uint32 {
	gen.lck.Lock()
	defer gen.lck.Unlock()
	n := gen.nextSerial
	for gen.serialUsed[n] {
		n++
	}
	gen.serialUsed[n] = true
	gen.nextSerial = n + 1
	return n
}

func (gen *serialGenerator) RetireSerial(serial uint32) {
	gen.lck.Lock()
	defer gen.lck.Unlock()
	delete(gen.serialUsed, serial)
}

type nameTracker struct {
	lck    sync.RWMutex
	unique string
	names  map[string]struct{}
}

func newNameTracker() *nameTracker {
	return &nameTracker{names: map[string]struct{}{}}
}
func (tracker *nameTracker) acquireUniqueConnectionName(name string) {
	tracker.lck.Lock()
	defer tracker.lck.Unlock()
	tracker.unique = name
}
func (tracker *nameTracker) acquireName(name string) {
	tracker.lck.Lock()
	defer tracker.lck.Unlock()
	tracker.names[name] = struct{}{}
}
func (tracker *nameTracker) loseName(name string) {
	tracker.lck.Lock()
	defer tracker.lck.Unlock()
	delete(tracker.names, name)
}

func (tracker *nameTracker) uniqueNameIsKnown() bool {
	tracker.lck.RLock()
	defer tracker.lck.RUnlock()
	return tracker.unique != ""
}
func (tracker *nameTracker) isKnownName(name string) bool {
	tracker.lck.RLock()
	defer tracker.lck.RUnlock()
	_, ok := tracker.names[name]
	return ok || name == tracker.unique
}
func (tracker *nameTracker) listKnownNames() []string {
	tracker.lck.RLock()
	defer tracker.lck.RUnlock()
	out := make([]string, 0, len(tracker.names)+1)
	out = append(out, tracker.unique)
	for k := range tracker.names {
		out = append(out, k)
	}
	return out
}

type callTracker struct {
	calls map[uint32]*Call
	lck   sync.RWMutex
}

func newCallTracker() *callTracker {
	return &callTracker{calls: map[uint32]*Call{}}
}

func (tracker *callTracker) track(sn uint32, call *Call) {
	tracker.lck.Lock()
	tracker.calls[sn] = call
	tracker.lck.Unlock()
}

func (tracker *callTracker) handleReply(msg *Message) uint32 {
	serial := msg.Headers[FieldReplySerial].value.(uint32)
	tracker.lck.RLock()
	_, ok := tracker.calls[serial]
	tracker.lck.RUnlock()
	if ok {
		tracker.finalizeWithBody(serial, msg.Body)
	}
	return serial
}

func (tracker *callTracker) handleDBusError(msg *Message) uint32 {
	serial := msg.Headers[FieldReplySerial].value.(uint32)
	tracker.lck.RLock()
	_, ok := tracker.calls[serial]
	tracker.lck.RUnlock()
	if ok {
		name, _ := msg.Headers[FieldErrorName].value.(string)
		tracker.finalizeWithError(serial, Error{name, msg.Body})
	}
	return serial
}

func (tracker *callTracker) handleSendError(msg *Message, err error) {
	if err == nil {
		return
	}
	tracker.lck.RLock()
	_, ok := tracker.calls[msg.serial]
	tracker.lck.RUnlock()
	if ok {
		tracker.finalizeWithError(msg.serial, err)
	}
}

// finalize was the only func that did not strobe Done
func (tracker *callTracker) finalize(sn uint32) {
	tracker.lck.Lock()
	defer tracker.lck.Unlock()
	c, ok := tracker.calls[sn]
	if ok {
		delete(tracker.calls, sn)
		c.ContextCancel()
	}
	return
}

func (tracker *callTracker) finalizeWithBody(sn uint32, body []interface{}) {
	tracker.lck.Lock()
	c, ok := tracker.calls[sn]
	if ok {
		delete(tracker.calls, sn)
	}
	tracker.lck.Unlock()
	if ok {
		c.Body = body
		c.done()
	}
	return
}

func (tracker *callTracker) finalizeWithError(sn uint32, err error) {
	tracker.lck.Lock()
	c, ok := tracker.calls[sn]
	if ok {
		delete(tracker.calls, sn)
	}
	tracker.lck.Unlock()
	if ok {
		c.Err = err
		c.done()
	}
	return
}

func (tracker *callTracker) finalizeAllWithError(err error) {
	tracker.lck.Lock()
	closedCalls := make([]*Call, 0, len(tracker.calls))
	for sn := range tracker.calls {
		closedCalls = append(closedCalls, tracker.calls[sn])
	}
	tracker.calls = map[uint32]*Call{}
	tracker.lck.Unlock()
	for _, call := range closedCalls {
		call.Err = err
		call.done()
	}
}
