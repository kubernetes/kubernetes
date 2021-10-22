package eventstream

import "fmt"

// LengthError provides the error for items being larger than a maximum length.
type LengthError struct {
	Part  string
	Want  int
	Have  int
	Value interface{}
}

func (e LengthError) Error() string {
	return fmt.Sprintf("%s length invalid, %d/%d, %v",
		e.Part, e.Want, e.Have, e.Value)
}

// ChecksumError provides the error for message checksum invalidation errors.
type ChecksumError struct{}

func (e ChecksumError) Error() string {
	return "message checksum mismatch"
}
