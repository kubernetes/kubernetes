package objx

// Value provides methods for extracting interface{} data in various
// types.
type Value struct {
	// data contains the raw data being managed by this Value
	data interface{}
}

// Data returns the raw data contained by this Value
func (v *Value) Data() interface{} {
	return v.data
}
