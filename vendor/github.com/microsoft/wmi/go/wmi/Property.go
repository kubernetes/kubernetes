package wmi

type PropertyFlags int

const (
	// None
	None PropertyFlags = 0
	// Class
	WClass PropertyFlags = 1
	// Method
	Method PropertyFlags = 2
	// Property
	WProperty PropertyFlags = 4
	// Parameter
	Parameter PropertyFlags = 8
	// Association
	Association PropertyFlags = 16
	// Indication
	Indication PropertyFlags = 32
	// Reference
	Reference PropertyFlags = 64
	// Any
	Any PropertyFlags = 127
	// EnableOverride
	EnableOverride PropertyFlags = 128
	// DisableOverride
	DisableOverride PropertyFlags = 256
	// Restricted
	Restricted PropertyFlags = 512
	// ToSubClass
	ToSubclass PropertyFlags = 1024
	// Translatable
	Translatable PropertyFlags = 2048
	// Key
	Key PropertyFlags = 4096
	// In
	In PropertyFlags = 8192
	// Out
	Out PropertyFlags = 16384
	// Required
	Required PropertyFlags = 32768
	// Static
	Static PropertyFlags = 65536
	// Abstract
	Abstract PropertyFlags = 131072
	// Terminal
	Terminal PropertyFlags = 262144
	// Expensive
	Expensive PropertyFlags = 524288
	// Stream
	Stream PropertyFlags = 1048576
	// ReadOnly
	ReadOnly PropertyFlags = 2097152
	// NotModified
	NotModified PropertyFlags = 33554432
	// NullValue
	NullValue PropertyFlags = 536870912
	// Borrow
	Borrow PropertyFlags = 1073741824
	// Adopt
	//Adopt PropertyFlags  = 2147483648;
)

type WmiType int

const (
	WbemCimtypeSint8     WmiType = 16
	WbemCimtypeUint8     WmiType = 17
	WbemCimtypeSint16    WmiType = 2
	WbemCimtypeUint16    WmiType = 18
	WbemCimtypeSint32    WmiType = 3
	WbemCimtypeUint32    WmiType = 19
	WbemCimtypeSint64    WmiType = 20
	WbemCimtypeUint64    WmiType = 21
	WbemCimtypeReal32    WmiType = 4
	WbemCimtypeReal64    WmiType = 5
	WbemCimtypeBoolean   WmiType = 11
	WbemCimtypeString    WmiType = 8
	WbemCimtypeDatetime  WmiType = 101
	WbemCimtypeReference WmiType = 102
	WbemCimtypeChar16    WmiType = 103
	WbemCimtypeObject    WmiType = 13
)

// Property
type Property interface {
	Name() string
	Value() string
	Type() WmiType
	Flags() PropertyFlags
}
