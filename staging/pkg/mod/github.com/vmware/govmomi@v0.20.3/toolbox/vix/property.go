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

package vix

import (
	"bytes"
	"encoding/binary"
	"errors"
)

// Property type enum as defined in open-vm-tools/lib/include/vix.h
const (
	_ = iota // ANY type not supported
	vixPropertyTypeInt32
	vixPropertyTypeString
	vixPropertyTypeBool
	_ // HANDLE type not supported
	vixPropertyTypeInt64
	vixPropertyTypeBlob
)

// Property ID enum as defined in open-vm-tools/lib/include/vixOpenSource.h
const (
	PropertyGuestToolsAPIOptions = 4501
	PropertyGuestOsFamily        = 4502
	PropertyGuestOsVersion       = 4503
	PropertyGuestToolsProductNam = 4511
	PropertyGuestToolsVersion    = 4500
	PropertyGuestName            = 4505
	PropertyGuestOsVersionShort  = 4520

	PropertyGuestStartProgramEnabled            = 4540
	PropertyGuestListProcessesEnabled           = 4541
	PropertyGuestTerminateProcessEnabled        = 4542
	PropertyGuestReadEnvironmentVariableEnabled = 4543

	PropertyGuestMakeDirectoryEnabled                 = 4547
	PropertyGuestDeleteFileEnabled                    = 4548
	PropertyGuestDeleteDirectoryEnabled               = 4549
	PropertyGuestMoveDirectoryEnabled                 = 4550
	PropertyGuestMoveFileEnabled                      = 4551
	PropertyGuestCreateTempFileEnabled                = 4552
	PropertyGuestCreateTempDirectoryEnabled           = 4553
	PropertyGuestListFilesEnabled                     = 4554
	PropertyGuestChangeFileAttributesEnabled          = 4555
	PropertyGuestInitiateFileTransferFromGuestEnabled = 4556
	PropertyGuestInitiateFileTransferToGuestEnabled   = 4557
)

type Property struct {
	header struct {
		ID     int32
		Kind   int32
		Length int32
	}

	data struct {
		Int32  int32
		String string
		Bool   uint8
		Int64  int64
		Blob   []byte
	}
}

var int32Size int32

func init() {
	var i int32
	int32Size = int32(binary.Size(&i))
}

type PropertyList []*Property

func NewInt32Property(ID int32, val int32) *Property {
	p := new(Property)
	p.header.ID = ID
	p.header.Kind = vixPropertyTypeInt32
	p.header.Length = int32Size
	p.data.Int32 = val
	return p
}

func NewStringProperty(ID int32, val string) *Property {
	p := new(Property)
	p.header.ID = ID
	p.header.Kind = vixPropertyTypeString
	p.header.Length = int32(len(val) + 1)
	p.data.String = val
	return p
}

func NewBoolProperty(ID int32, val bool) *Property {
	p := new(Property)
	p.header.ID = ID
	p.header.Kind = vixPropertyTypeBool
	p.header.Length = 1
	if val {
		p.data.Bool = 1
	}
	return p
}

func NewInt64Property(ID int32, val int64) *Property {
	p := new(Property)
	p.header.ID = ID
	p.header.Kind = vixPropertyTypeInt64
	p.header.Length = int32Size * 2
	p.data.Int64 = val
	return p
}

func NewBlobProperty(ID int32, val []byte) *Property {
	p := new(Property)
	p.header.ID = ID
	p.header.Kind = vixPropertyTypeBlob
	p.header.Length = int32(len(val))
	p.data.Blob = val
	return p
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (p *Property) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	// #nosec: Errors unhandled
	_ = binary.Write(buf, binary.LittleEndian, &p.header)

	switch p.header.Kind {
	case vixPropertyTypeBool:
		// #nosec: Errors unhandled
		_ = binary.Write(buf, binary.LittleEndian, p.data.Bool)
	case vixPropertyTypeInt32:
		// #nosec: Errors unhandled
		_ = binary.Write(buf, binary.LittleEndian, p.data.Int32)
	case vixPropertyTypeInt64:
		// #nosec: Errors unhandled
		_ = binary.Write(buf, binary.LittleEndian, p.data.Int64)
	case vixPropertyTypeString:
		// #nosec: Errors unhandled
		_, _ = buf.WriteString(p.data.String)
		// #nosec: Errors unhandled
		_ = buf.WriteByte(0)
	case vixPropertyTypeBlob:
		// #nosec: Errors unhandled
		_, _ = buf.Write(p.data.Blob)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (p *Property) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &p.header)
	if err != nil {
		return err
	}

	switch p.header.Kind {
	case vixPropertyTypeBool:
		return binary.Read(buf, binary.LittleEndian, &p.data.Bool)
	case vixPropertyTypeInt32:
		return binary.Read(buf, binary.LittleEndian, &p.data.Int32)
	case vixPropertyTypeInt64:
		return binary.Read(buf, binary.LittleEndian, &p.data.Int64)
	case vixPropertyTypeString:
		s := make([]byte, p.header.Length)
		if _, err := buf.Read(s); err != nil {
			return err
		}

		p.data.String = string(bytes.TrimRight(s, "\x00"))
	case vixPropertyTypeBlob:
		p.data.Blob = make([]byte, p.header.Length)
		if _, err := buf.Read(p.data.Blob); err != nil {
			return err
		}
	default:
		return errors.New("VIX_E_UNRECOGNIZED_PROPERTY")
	}

	return nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (l *PropertyList) UnmarshalBinary(data []byte) error {
	headerSize := int32Size * 3

	for {
		p := new(Property)

		err := p.UnmarshalBinary(data)
		if err != nil {
			return err
		}

		*l = append(*l, p)

		offset := headerSize + p.header.Length
		data = data[offset:]

		if len(data) == 0 {
			return nil
		}
	}
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (l *PropertyList) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer

	for _, p := range *l {
		// #nosec: Errors unhandled
		b, _ := p.MarshalBinary()
		// #nosec: Errors unhandled
		_, _ = buf.Write(b)
	}

	return buf.Bytes(), nil
}
