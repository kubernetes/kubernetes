package jsonpath

import (
	"encoding/json"
	"io"
)

// KeyString is returned from Decoder.Token to represent each key in a JSON object value.
type KeyString string

// Decoder extends the Go runtime's encoding/json.Decoder to support navigating in a stream of JSON tokens.
type Decoder struct {
	json.Decoder

	path    JsonPath
	context jsonContext
}

// NewDecoder creates a new instance of the extended JSON Decoder.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{Decoder: *json.NewDecoder(r)}
}

// SeekTo causes the Decoder to move forward to a given path in the JSON structure.
//
// The path argument must consist of strings or integers. Each string specifies an JSON object key, and
// each integer specifies an index into a JSON array.
//
// Consider the JSON structure
//
//  { "a": [0,"s",12e4,{"b":0,"v":35} ] }
//
// SeekTo("a",3,"v") will move to the value referenced by the "a" key in the current object,
// followed by a move to the 4th value (index 3) in the array, followed by a move to the value at key "v".
// In this example, a subsequent call to the decoder's Decode() would unmarshal the value 35.
//
// SeekTo returns a boolean value indicating whether a match was found.
//
// Decoder is intended to be used with a stream of tokens. As a result it navigates forward only.
func (d *Decoder) SeekTo(path ...interface{}) (bool, error) {

	if len(path) > 0 {
		last := len(path) - 1
		if i, ok := path[last].(int); ok {
			path[last] = i - 1
		}
	}

	for {
		if len(path) == len(d.path) && d.path.Equal(path) {
			return true, nil
		}
		_, err := d.Token()
		if err == io.EOF {
			return false, nil
		} else if err != nil {
			return false, err
		}
	}
}

// Decode reads the next JSON-encoded value from its input and stores it in the value pointed to by v. This is
// equivalent to encoding/json.Decode().
func (d *Decoder) Decode(v interface{}) error {
	switch d.context {
	case objValue:
		d.context = objKey
		break
	case arrValue:
		d.path.incTop()
		break
	}
	return d.Decoder.Decode(v)
}

// Path returns a slice of string and/or int values representing the path from the root of the JSON object to the
// position of the most-recently parsed token.
func (d *Decoder) Path() JsonPath {
	p := make(JsonPath, len(d.path))
	copy(p, d.path)
	return p
}

// Token is equivalent to the Token() method on json.Decoder. The primary difference is that it distinguishes
// between strings that are keys and and strings that are values. String tokens that are object keys are returned as a
// KeyString rather than as a native string.
func (d *Decoder) Token() (json.Token, error) {
	t, err := d.Decoder.Token()
	if err != nil {
		return t, err
	}

	if t == nil {
		switch d.context {
		case objValue:
			d.context = objKey
			break
		case arrValue:
			d.path.incTop()
			break
		}
		return t, err
	}

	switch t := t.(type) {
	case json.Delim:
		switch t {
		case json.Delim('{'):
			if d.context == arrValue {
				d.path.incTop()
			}
			d.path.push("")
			d.context = objKey
			break
		case json.Delim('}'):
			d.path.pop()
			d.context = d.path.inferContext()
			break
		case json.Delim('['):
			if d.context == arrValue {
				d.path.incTop()
			}
			d.path.push(-1)
			d.context = arrValue
			break
		case json.Delim(']'):
			d.path.pop()
			d.context = d.path.inferContext()
			break
		}
	case float64, json.Number, bool:
		switch d.context {
		case objValue:
			d.context = objKey
			break
		case arrValue:
			d.path.incTop()
			break
		}
		break
	case string:
		switch d.context {
		case objKey:
			d.path.nameTop(t)
			d.context = objValue
			return KeyString(t), err
		case objValue:
			d.context = objKey
		case arrValue:
			d.path.incTop()
		}
		break
	}

	return t, err
}

// Scan moves forward over the JSON stream consuming all the tokens at the current level (current object, current array)
// invoking each matching PathAction along the way.
//
// Scan returns true if there are more contiguous values to scan (for example in an array).
func (d *Decoder) Scan(ext *PathActions) (bool, error) {

	rootPath := d.Path()

	// If this is an array path, increment the root path in our local copy.
	if rootPath.inferContext() == arrValue {
		rootPath.incTop()
	}

	for {
		// advance the token position
		_, err := d.Token()
		if err != nil {
			return false, err
		}

	match:
		var relPath JsonPath

		// capture the new JSON path
		path := d.Path()

		if len(path) > len(rootPath) {
			// capture the path relative to where the scan started
			relPath = path[len(rootPath):]
		} else {
			// if the path is not longer than the root, then we are done with this scan
			// return boolean flag indicating if there are more items to scan at the same level
			return d.Decoder.More(), nil
		}

		// match the relative path against the path actions
		if node := ext.node.match(relPath); node != nil {
			if node.action != nil {
				// we have a match so execute the action
				err = node.action(d)
				if err != nil {
					return d.Decoder.More(), err
				}
				// The action may have advanced the decoder. If we are in an array, advancing it further would
				// skip tokens. So, if we are scanning an array, jump to the top without advancing the token.
				if d.path.inferContext() == arrValue && d.Decoder.More() {
					goto match
				}
			}
		}
	}
}
