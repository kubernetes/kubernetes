package winio

import (
	"bytes"
	"encoding/binary"
	"errors"
)

type fileFullEaInformation struct {
	NextEntryOffset uint32
	Flags           uint8
	NameLength      uint8
	ValueLength     uint16
}

var (
	fileFullEaInformationSize = binary.Size(&fileFullEaInformation{})

	errInvalidEaBuffer = errors.New("invalid extended attribute buffer")
	errEaNameTooLarge  = errors.New("extended attribute name too large")
	errEaValueTooLarge = errors.New("extended attribute value too large")
)

// ExtendedAttribute represents a single Windows EA.
type ExtendedAttribute struct {
	Name  string
	Value []byte
	Flags uint8
}

func parseEa(b []byte) (ea ExtendedAttribute, nb []byte, err error) {
	var info fileFullEaInformation
	err = binary.Read(bytes.NewReader(b), binary.LittleEndian, &info)
	if err != nil {
		err = errInvalidEaBuffer
		return
	}

	nameOffset := fileFullEaInformationSize
	nameLen := int(info.NameLength)
	valueOffset := nameOffset + int(info.NameLength) + 1
	valueLen := int(info.ValueLength)
	nextOffset := int(info.NextEntryOffset)
	if valueLen+valueOffset > len(b) || nextOffset < 0 || nextOffset > len(b) {
		err = errInvalidEaBuffer
		return
	}

	ea.Name = string(b[nameOffset : nameOffset+nameLen])
	ea.Value = b[valueOffset : valueOffset+valueLen]
	ea.Flags = info.Flags
	if info.NextEntryOffset != 0 {
		nb = b[info.NextEntryOffset:]
	}
	return
}

// DecodeExtendedAttributes decodes a list of EAs from a FILE_FULL_EA_INFORMATION
// buffer retrieved from BackupRead, ZwQueryEaFile, etc.
func DecodeExtendedAttributes(b []byte) (eas []ExtendedAttribute, err error) {
	for len(b) != 0 {
		ea, nb, err := parseEa(b)
		if err != nil {
			return nil, err
		}

		eas = append(eas, ea)
		b = nb
	}
	return
}

func writeEa(buf *bytes.Buffer, ea *ExtendedAttribute, last bool) error {
	if int(uint8(len(ea.Name))) != len(ea.Name) {
		return errEaNameTooLarge
	}
	if int(uint16(len(ea.Value))) != len(ea.Value) {
		return errEaValueTooLarge
	}
	entrySize := uint32(fileFullEaInformationSize + len(ea.Name) + 1 + len(ea.Value))
	withPadding := (entrySize + 3) &^ 3
	nextOffset := uint32(0)
	if !last {
		nextOffset = withPadding
	}
	info := fileFullEaInformation{
		NextEntryOffset: nextOffset,
		Flags:           ea.Flags,
		NameLength:      uint8(len(ea.Name)),
		ValueLength:     uint16(len(ea.Value)),
	}

	err := binary.Write(buf, binary.LittleEndian, &info)
	if err != nil {
		return err
	}

	_, err = buf.Write([]byte(ea.Name))
	if err != nil {
		return err
	}

	err = buf.WriteByte(0)
	if err != nil {
		return err
	}

	_, err = buf.Write(ea.Value)
	if err != nil {
		return err
	}

	_, err = buf.Write([]byte{0, 0, 0}[0 : withPadding-entrySize])
	if err != nil {
		return err
	}

	return nil
}

// EncodeExtendedAttributes encodes a list of EAs into a FILE_FULL_EA_INFORMATION
// buffer for use with BackupWrite, ZwSetEaFile, etc.
func EncodeExtendedAttributes(eas []ExtendedAttribute) ([]byte, error) {
	var buf bytes.Buffer
	for i := range eas {
		last := false
		if i == len(eas)-1 {
			last = true
		}

		err := writeEa(&buf, &eas[i], last)
		if err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}
