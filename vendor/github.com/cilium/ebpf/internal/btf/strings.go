package btf

import (
	"bytes"
	"io"
	"io/ioutil"

	"golang.org/x/xerrors"
)

type stringTable []byte

func readStringTable(r io.Reader) (stringTable, error) {
	contents, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, xerrors.Errorf("can't read string table: %v", err)
	}

	if len(contents) < 1 {
		return nil, xerrors.New("string table is empty")
	}

	if contents[0] != '\x00' {
		return nil, xerrors.New("first item in string table is non-empty")
	}

	if contents[len(contents)-1] != '\x00' {
		return nil, xerrors.New("string table isn't null terminated")
	}

	return stringTable(contents), nil
}

func (st stringTable) Lookup(offset uint32) (string, error) {
	if int64(offset) > int64(^uint(0)>>1) {
		return "", xerrors.Errorf("offset %d overflows int", offset)
	}

	pos := int(offset)
	if pos >= len(st) {
		return "", xerrors.Errorf("offset %d is out of bounds", offset)
	}

	if pos > 0 && st[pos-1] != '\x00' {
		return "", xerrors.Errorf("offset %d isn't start of a string", offset)
	}

	str := st[pos:]
	end := bytes.IndexByte(str, '\x00')
	if end == -1 {
		return "", xerrors.Errorf("offset %d isn't null terminated", offset)
	}

	return string(str[:end]), nil
}

func (st stringTable) LookupName(offset uint32) (Name, error) {
	str, err := st.Lookup(offset)
	return Name(str), err
}
