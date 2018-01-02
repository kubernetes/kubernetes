//+build jsoniter-sloppy

package jsoniter

import (
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func Test_string_end(t *testing.T) {
	end, escaped := ParseString(ConfigDefault, `abc"`).findStringEnd()
	if end != 4 {
		t.Fatal(end)
	}
	if escaped != false {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `abc\\"`).findStringEnd()
	if end != 6 {
		t.Fatal(end)
	}
	if escaped != true {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `abc\\\\"`).findStringEnd()
	if end != 8 {
		t.Fatal(end)
	}
	if escaped != true {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `abc\"`).findStringEnd()
	if end != -1 {
		t.Fatal(end)
	}
	if escaped != false {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `abc\`).findStringEnd()
	if end != -1 {
		t.Fatal(end)
	}
	if escaped != true {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `abc\\`).findStringEnd()
	if end != -1 {
		t.Fatal(end)
	}
	if escaped != false {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `\\`).findStringEnd()
	if end != -1 {
		t.Fatal(end)
	}
	if escaped != false {
		t.Fatal(escaped)
	}
	end, escaped = ParseString(ConfigDefault, `\`).findStringEnd()
	if end != -1 {
		t.Fatal(end)
	}
	if escaped != true {
		t.Fatal(escaped)
	}
}

type StagedReader struct {
	r1 string
	r2 string
	r3 string
	r  int
}

func (reader *StagedReader) Read(p []byte) (n int, err error) {
	reader.r++
	switch reader.r {
	case 1:
		copy(p, []byte(reader.r1))
		return len(reader.r1), nil
	case 2:
		copy(p, []byte(reader.r2))
		return len(reader.r2), nil
	case 3:
		copy(p, []byte(reader.r3))
		return len(reader.r3), nil
	default:
		return 0, io.EOF
	}
}

func Test_skip_string(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `"abc`)
	iter.skipString()
	should.Equal(1, iter.head)
	iter = ParseString(ConfigDefault, `\""abc`)
	iter.skipString()
	should.Equal(3, iter.head)
	reader := &StagedReader{
		r1: `abc`,
		r2: `"`,
	}
	iter = Parse(ConfigDefault, reader, 4096)
	iter.skipString()
	should.Equal(1, iter.head)
	reader = &StagedReader{
		r1: `abc`,
		r2: `1"`,
	}
	iter = Parse(ConfigDefault, reader, 4096)
	iter.skipString()
	should.Equal(2, iter.head)
	reader = &StagedReader{
		r1: `abc\`,
		r2: `"`,
	}
	iter = Parse(ConfigDefault, reader, 4096)
	iter.skipString()
	should.NotNil(iter.Error)
	reader = &StagedReader{
		r1: `abc\`,
		r2: `""`,
	}
	iter = Parse(ConfigDefault, reader, 4096)
	iter.skipString()
	should.Equal(2, iter.head)
}

func Test_skip_object(t *testing.T) {
	iter := ParseString(ConfigDefault, `}`)
	iter.skipObject()
	if iter.head != 1 {
		t.Fatal(iter.head)
	}
	iter = ParseString(ConfigDefault, `a}`)
	iter.skipObject()
	if iter.head != 2 {
		t.Fatal(iter.head)
	}
	iter = ParseString(ConfigDefault, `{}}a`)
	iter.skipObject()
	if iter.head != 3 {
		t.Fatal(iter.head)
	}
	reader := &StagedReader{
		r1: `{`,
		r2: `}}a`,
	}
	iter = Parse(ConfigDefault, reader, 4096)
	iter.skipObject()
	if iter.head != 2 {
		t.Fatal(iter.head)
	}
	iter = ParseString(ConfigDefault, `"}"}a`)
	iter.skipObject()
	if iter.head != 4 {
		t.Fatal(iter.head)
	}
}
