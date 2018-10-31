package eventstream

import (
	"encoding/binary"
	"fmt"
	"io"
)

// Headers are a collection of EventStream header values.
type Headers []Header

// Header is a single EventStream Key Value header pair.
type Header struct {
	Name  string
	Value Value
}

// Set associates the name with a value. If the header name already exists in
// the Headers the value will be replaced with the new one.
func (hs *Headers) Set(name string, value Value) {
	var i int
	for ; i < len(*hs); i++ {
		if (*hs)[i].Name == name {
			(*hs)[i].Value = value
			return
		}
	}

	*hs = append(*hs, Header{
		Name: name, Value: value,
	})
}

// Get returns the Value associated with the header. Nil is returned if the
// value does not exist.
func (hs Headers) Get(name string) Value {
	for i := 0; i < len(hs); i++ {
		if h := hs[i]; h.Name == name {
			return h.Value
		}
	}
	return nil
}

// Del deletes the value in the Headers if it exists.
func (hs *Headers) Del(name string) {
	for i := 0; i < len(*hs); i++ {
		if (*hs)[i].Name == name {
			copy((*hs)[i:], (*hs)[i+1:])
			(*hs) = (*hs)[:len(*hs)-1]
		}
	}
}

func decodeHeaders(r io.Reader) (Headers, error) {
	hs := Headers{}

	for {
		name, err := decodeHeaderName(r)
		if err != nil {
			if err == io.EOF {
				// EOF while getting header name means no more headers
				break
			}
			return nil, err
		}

		value, err := decodeHeaderValue(r)
		if err != nil {
			return nil, err
		}

		hs.Set(name, value)
	}

	return hs, nil
}

func decodeHeaderName(r io.Reader) (string, error) {
	var n headerName

	var err error
	n.Len, err = decodeUint8(r)
	if err != nil {
		return "", err
	}

	name := n.Name[:n.Len]
	if _, err := io.ReadFull(r, name); err != nil {
		return "", err
	}

	return string(name), nil
}

func decodeHeaderValue(r io.Reader) (Value, error) {
	var raw rawValue

	typ, err := decodeUint8(r)
	if err != nil {
		return nil, err
	}
	raw.Type = valueType(typ)

	var v Value

	switch raw.Type {
	case trueValueType:
		v = BoolValue(true)
	case falseValueType:
		v = BoolValue(false)
	case int8ValueType:
		var tv Int8Value
		err = tv.decode(r)
		v = tv
	case int16ValueType:
		var tv Int16Value
		err = tv.decode(r)
		v = tv
	case int32ValueType:
		var tv Int32Value
		err = tv.decode(r)
		v = tv
	case int64ValueType:
		var tv Int64Value
		err = tv.decode(r)
		v = tv
	case bytesValueType:
		var tv BytesValue
		err = tv.decode(r)
		v = tv
	case stringValueType:
		var tv StringValue
		err = tv.decode(r)
		v = tv
	case timestampValueType:
		var tv TimestampValue
		err = tv.decode(r)
		v = tv
	case uuidValueType:
		var tv UUIDValue
		err = tv.decode(r)
		v = tv
	default:
		panic(fmt.Sprintf("unknown value type %d", raw.Type))
	}

	// Error could be EOF, let caller deal with it
	return v, err
}

const maxHeaderNameLen = 255

type headerName struct {
	Len  uint8
	Name [maxHeaderNameLen]byte
}

func (v headerName) encode(w io.Writer) error {
	if err := binary.Write(w, binary.BigEndian, v.Len); err != nil {
		return err
	}

	_, err := w.Write(v.Name[:v.Len])
	return err
}
