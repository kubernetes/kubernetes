package jsoniter

import (
	"bytes"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func Test_read_by_one(t *testing.T) {
	iter := Parse(ConfigDefault, bytes.NewBufferString("abc"), 1)
	b := iter.readByte()
	if iter.Error != nil {
		t.Fatal(iter.Error)
	}
	if b != 'a' {
		t.Fatal(b)
	}
	iter.unreadByte()
	if iter.Error != nil {
		t.Fatal(iter.Error)
	}
	b = iter.readByte()
	if iter.Error != nil {
		t.Fatal(iter.Error)
	}
	if b != 'a' {
		t.Fatal(b)
	}
}

func Test_read_by_two(t *testing.T) {
	should := require.New(t)
	iter := Parse(ConfigDefault, bytes.NewBufferString("abc"), 2)
	b := iter.readByte()
	should.Nil(iter.Error)
	should.Equal(byte('a'), b)
	b = iter.readByte()
	should.Nil(iter.Error)
	should.Equal(byte('b'), b)
	iter.unreadByte()
	should.Nil(iter.Error)
	iter.unreadByte()
	should.Nil(iter.Error)
	b = iter.readByte()
	should.Nil(iter.Error)
	should.Equal(byte('a'), b)
}

func Test_read_until_eof(t *testing.T) {
	iter := Parse(ConfigDefault, bytes.NewBufferString("abc"), 2)
	iter.readByte()
	iter.readByte()
	b := iter.readByte()
	if iter.Error != nil {
		t.Fatal(iter.Error)
	}
	if b != 'c' {
		t.Fatal(b)
	}
	iter.readByte()
	if iter.Error != io.EOF {
		t.Fatal(iter.Error)
	}
}
