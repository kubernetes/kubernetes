package pflag

import (
	"encoding/hex"
	"fmt"
	"strings"
)

// BytesHex adapts []byte for use as a flag. Value of flag is HEX encoded
type bytesHexValue []byte

func (bytesHex bytesHexValue) String() string {
	return fmt.Sprintf("%X", []byte(bytesHex))
}

func (bytesHex *bytesHexValue) Set(value string) error {
	bin, err := hex.DecodeString(strings.TrimSpace(value))

	if err != nil {
		return err
	}

	*bytesHex = bin

	return nil
}

func (*bytesHexValue) Type() string {
	return "bytesHex"
}

func newBytesHexValue(val []byte, p *[]byte) *bytesHexValue {
	*p = val
	return (*bytesHexValue)(p)
}

func bytesHexConv(sval string) (interface{}, error) {

	bin, err := hex.DecodeString(sval)

	if err == nil {
		return bin, nil
	}

	return nil, fmt.Errorf("invalid string being converted to Bytes: %s %s", sval, err)
}

// GetBytesHex return the []byte value of a flag with the given name
func (f *FlagSet) GetBytesHex(name string) ([]byte, error) {
	val, err := f.getFlagType(name, "bytesHex", bytesHexConv)

	if err != nil {
		return []byte{}, err
	}

	return val.([]byte), nil
}

// BytesHexVar defines an []byte flag with specified name, default value, and usage string.
// The argument p points to an []byte variable in which to store the value of the flag.
func (f *FlagSet) BytesHexVar(p *[]byte, name string, value []byte, usage string) {
	f.VarP(newBytesHexValue(value, p), name, "", usage)
}

// BytesHexVarP is like BytesHexVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) BytesHexVarP(p *[]byte, name, shorthand string, value []byte, usage string) {
	f.VarP(newBytesHexValue(value, p), name, shorthand, usage)
}

// BytesHexVar defines an []byte flag with specified name, default value, and usage string.
// The argument p points to an []byte variable in which to store the value of the flag.
func BytesHexVar(p *[]byte, name string, value []byte, usage string) {
	CommandLine.VarP(newBytesHexValue(value, p), name, "", usage)
}

// BytesHexVarP is like BytesHexVar, but accepts a shorthand letter that can be used after a single dash.
func BytesHexVarP(p *[]byte, name, shorthand string, value []byte, usage string) {
	CommandLine.VarP(newBytesHexValue(value, p), name, shorthand, usage)
}

// BytesHex defines an []byte flag with specified name, default value, and usage string.
// The return value is the address of an []byte variable that stores the value of the flag.
func (f *FlagSet) BytesHex(name string, value []byte, usage string) *[]byte {
	p := new([]byte)
	f.BytesHexVarP(p, name, "", value, usage)
	return p
}

// BytesHexP is like BytesHex, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) BytesHexP(name, shorthand string, value []byte, usage string) *[]byte {
	p := new([]byte)
	f.BytesHexVarP(p, name, shorthand, value, usage)
	return p
}

// BytesHex defines an []byte flag with specified name, default value, and usage string.
// The return value is the address of an []byte variable that stores the value of the flag.
func BytesHex(name string, value []byte, usage string) *[]byte {
	return CommandLine.BytesHexP(name, "", value, usage)
}

// BytesHexP is like BytesHex, but accepts a shorthand letter that can be used after a single dash.
func BytesHexP(name, shorthand string, value []byte, usage string) *[]byte {
	return CommandLine.BytesHexP(name, shorthand, value, usage)
}
