package json

// Marshaler matches the standard library interface to allow types such as
// RawMessage to advertise custom JSON encoding behavior without requiring
// the full encoding stack.
type Marshaler interface {
	MarshalJSON() ([]byte, error)
}
