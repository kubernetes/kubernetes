/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package hgfs

import (
	"bytes"
	"encoding"
	"encoding/binary"
)

// MarshalBinary is a wrapper around binary.Write
func MarshalBinary(fields ...interface{}) ([]byte, error) {
	buf := new(bytes.Buffer)

	for _, p := range fields {
		switch m := p.(type) {
		case encoding.BinaryMarshaler:
			data, err := m.MarshalBinary()
			if err != nil {
				return nil, ProtocolError(err)
			}

			_, _ = buf.Write(data)
		case []byte:
			_, _ = buf.Write(m)
		case string:
			_, _ = buf.WriteString(m)
		default:
			err := binary.Write(buf, binary.LittleEndian, p)
			if err != nil {
				return nil, ProtocolError(err)
			}
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary is a wrapper around binary.Read
func UnmarshalBinary(data []byte, fields ...interface{}) error {
	buf := bytes.NewBuffer(data)

	for _, p := range fields {
		switch m := p.(type) {
		case encoding.BinaryUnmarshaler:
			return m.UnmarshalBinary(buf.Bytes())
		case *[]byte:
			*m = buf.Bytes()
			return nil
		default:
			err := binary.Read(buf, binary.LittleEndian, p)
			if err != nil {
				return ProtocolError(err)
			}
		}
	}

	return nil
}
