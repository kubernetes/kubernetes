package dbus

// Terminator allows a handler to implement a shutdown mechanism that
// is called when the connection terminates.
type Terminator interface {
	Terminate()
}

// Handler is the representation of a D-Bus Application.
//
// The Handler must have a way to lookup objects given
// an ObjectPath. The returned object must implement the
// ServerObject interface.
type Handler interface {
	LookupObject(path ObjectPath) (ServerObject, bool)
}

// ServerObject is the representation of an D-Bus Object.
//
// Objects are registered at a path for a given Handler.
// The Objects implement D-Bus interfaces. The semantics
// of Interface lookup is up to the implementation of
// the ServerObject. The ServerObject implementation may
// choose to implement empty string as a valid interface
// represeting all methods or not per the D-Bus specification.
type ServerObject interface {
	LookupInterface(name string) (Interface, bool)
}

// An Interface is the representation of a D-Bus Interface.
//
// Interfaces are a grouping of methods implemented by the Objects.
// Interfaces are responsible for routing method calls.
type Interface interface {
	LookupMethod(name string) (Method, bool)
}

// A Method represents the exposed methods on D-Bus.
type Method interface {
	// Call requires that all arguments are decoded before being passed to it.
	Call(args ...interface{}) ([]interface{}, error)
	NumArguments() int
	NumReturns() int
	// ArgumentValue returns a representative value for the argument at position
	// it should be of the proper type. reflect.Zero would be a good mechanism
	// to use for this Value.
	ArgumentValue(position int) interface{}
	// ReturnValue returns a representative value for the return at position
	// it should be of the proper type. reflect.Zero would be a good mechanism
	// to use for this Value.
	ReturnValue(position int) interface{}
}

// An Argument Decoder can decode arguments using the non-standard mechanism
//
// If a method implements this interface then the non-standard
// decoder will be used.
//
// Method arguments must be decoded from the message.
// The mechanism for doing this will vary based on the
// implementation of the method. A normal approach is provided
// as part of this library, but may be replaced with
// any other decoding scheme.
type ArgumentDecoder interface {
	// To decode the arguments of a method the sender and message are
	// provided in case the semantics of the implementer provides access
	// to these as part of the method invocation.
	DecodeArguments(conn *Conn, sender string, msg *Message, args []interface{}) ([]interface{}, error)
}

// A SignalHandler is responsible for delivering a signal.
//
// Signal delivery may be changed from the default channel
// based approach by Handlers implementing the SignalHandler
// interface.
type SignalHandler interface {
	DeliverSignal(iface, name string, signal *Signal)
}

// SignalRegistrar manages signal delivery channels.
//
// This is an optional set of methods for `SignalHandler`.
type SignalRegistrar interface {
	AddSignal(ch chan<- *Signal)
	RemoveSignal(ch chan<- *Signal)
}

// A DBusError is used to convert a generic object to a D-Bus error.
//
// Any custom error mechanism may implement this interface to provide
// a custom encoding of the error on D-Bus. By default if a normal
// error is returned, it will be encoded as the generic
// "org.freedesktop.DBus.Error.Failed" error. By implementing this
// interface as well a custom encoding may be provided.
type DBusError interface {
	DBusError() (string, []interface{})
}

// SerialGenerator is responsible for serials generation.
//
// Different approaches for the serial generation can be used,
// maintaining a map guarded with a mutex (the standard way) or
// simply increment an atomic counter.
type SerialGenerator interface {
	GetSerial() uint32
	RetireSerial(serial uint32)
}
