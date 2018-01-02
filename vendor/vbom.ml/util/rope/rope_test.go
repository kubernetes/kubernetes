package rope

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/bruth/assert"
)

func TestEmptyRope(t *testing.T) {
	for _, r := range []Rope{Rope{}, New("")} {
		assert.Equal(t, int64(0), r.Len())

		assert.Equal(t, nil, r.Bytes())
		assert.Equal(t, "", r.String())

		assert.Equal(t, "", r.DropPrefix(3).String())
		assert.Equal(t, "", r.DropPrefix(-1).String())
		assert.Equal(t, "", r.DropPostfix(3).String())
		assert.Equal(t, "", r.DropPostfix(-1).String())

		assert.Equal(t, "", r.Slice(-1, 200).String())
		assert.Equal(t, "", r.Slice(0, 1).String())

		buf := bytes.NewBuffer(nil)
		_, _ = r.WriteTo(buf)
		assert.Equal(t, 0, buf.Len())
	}
}

func TestNew(t *testing.T) {
	for _, str := range []string{"", "abc"} {
		r := New(str)
		assert.Equal(t, r.node, leaf(str))
	}
}

func TestAppend(t *testing.T) {
	defer disableCoalesce()()

	r := New("123")
	assert.Equal(t, "123", r.String())

	assert.Equal(t, r, r.Append(Rope{}))
	assert.Equal(t, r, r.Append(New("")))

	// Test for structural equality in presence of empty strings.
	rab := r.Append(New("a"), New("c"), New("b"))
	raeb := r.Append(New("a"), New("c"), New(""), New("b"))
	assert.Equal(t, rab, raeb, "should ignore empty arguments to Append")

	r2 := r.Append(New("456"))
	assert.Equal(t, "123456", r2.String())
	assert.Equal(t, "123", r.String())

	r2 = r.Append(New("456"), New("abc"), New("def"))
	assert.Equal(t, "123456abcdef", r2.String())
	assert.Equal(t, "123", r.String())
}

func TestAppendString(t *testing.T) {
	defer disableCoalesce()()

	r := New("123")
	assert.Equal(t, "123", r.String())

	assert.Equal(t, r, r.Append(Rope{}))
	assert.Equal(t, r, r.Append(New("")))

	// Test for structural equality in presence of empty strings.
	rab := r.AppendString("a", "c", "b")
	raeb := r.AppendString("a", "c", "", "b")
	assert.Equal(t, rab, raeb, "should ignore empty arguments to AppendString")

	r2 := r.AppendString("456")
	assert.Equal(t, "123456", r2.String())
	assert.Equal(t, "123", r.String())

	r2 = r.AppendString("456", "abc", "def")
	assert.Equal(t, "123456abcdef", r2.String())
	assert.Equal(t, "123", r.String())
}

func TestRepeat(t *testing.T) {
	r := New("a")
	assert.Equal(t, "", r.Repeat(0).String())
	assert.Equal(t, "a", r.Repeat(1).String())
	assert.Equal(t, "aa", r.Repeat(2).String())
	assert.Equal(t, "aaa", r.Repeat(3).String())
	assert.Equal(t, "aaaa", r.Repeat(4).String())
	assert.Equal(t, "aaaaa", r.Repeat(5).String())
	assert.Equal(t, "aaaaaa", r.Repeat(6).String())
}

var treeR Rope

func init() {
	defer disableCoalesce()()

	treeR = New("123").AppendString("456", "abc").AppendString("def")
}

func TestAt(t *testing.T) {
	str := treeR.String()
	length := treeR.Len()
	for i := int64(0); i < length; i++ {
		assert.Equal(t, str[i], treeR.At(i))
	}
}

func TestLen(t *testing.T) {
	assert.Equal(t, int64(0), Rope{}.Len())
	assert.Equal(t, int64(12), treeR.Len())
}

func TestString(t *testing.T) {
	assert.Equal(t, "", Rope{}.String())
	assert.Equal(t, "123456abcdef", treeR.String())
}

func TestBytes(t *testing.T) {
	assert.Equal(t, []byte(nil), Rope{}.Bytes())
	assert.Equal(t, []byte(nil), New("").Bytes())
	assert.Equal(t, []byte("123456abcdef"), treeR.Bytes())
}

func TestWriteTo(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	_, _ = treeR.WriteTo(buf)

	assert.Equal(t, "123456abcdef", buf.String())
}

func TestSlice(t *testing.T) {
	defer disableCoalesce()()

	// See concat_test.go for the table used.
	for _, ss := range substrings {
		orig := Rope{ss.orig}
		got := orig.Slice(ss.start, ss.end)
		msg := fmt.Sprintf("%q[%v:%v] != %q", orig, ss.start, ss.end, got)
		assert.Equal(t, ss.want, got.node, msg)
	}
}

func TestDropPrefix(t *testing.T) {
	defer disableCoalesce()()

	// See concat_test.go for the table used.
	for _, ss := range substrings {
		if ss.end < ss.orig.length() {
			// Ignore non-suffix substrings
			continue
		}
		orig := Rope{ss.orig}
		got := orig.DropPrefix(ss.start)
		msg := fmt.Sprintf("%q[%v:] != %q", orig, ss.start, got)
		assert.Equal(t, ss.want, got.node, msg)
	}
}

func TestDropPostfix(t *testing.T) {
	defer disableCoalesce()()

	// See concat_test.go for the table used.
	for _, ss := range substrings {
		if ss.start > 0 {
			// Ignore non-prefix substrings
			continue
		}
		orig := Rope{ss.orig}
		got := orig.DropPostfix(ss.end)
		msg := fmt.Sprintf("%q[:%v] != %q", orig, ss.end, got)
		assert.Equal(t, ss.want, got.node, msg)
	}
}

func TestGoString(t *testing.T) {
	for i, format := range []string{"%v", "%#v"} {
		for _, str := range []string{"abc", "\""} {
			want := fmt.Sprintf(format, str)
			if MarkGoStringedRope && i == 1 {
				// GoStringer
				want = "/*Rope*/ " + want
			}
			assert.Equal(t, want, fmt.Sprintf(format, New(str)))
		}
	}
}

func TestWalk(t *testing.T) {
	defer disableCoalesce()()

	for _, r := range []Rope{Rope{}, emptyRope} {
		_ = r.Walk(func(_ string) error {
			t.Error("call to empty Rope's Walk parameter")
			return nil
		})
	}

	for _, r := range []Rope{
		New("abc").AppendString("def").AppendString("ghi"),
	} {
		str := r.String()
		err := r.Walk(func(part string) error {
			assert.Equal(t, str[:len(part)], part)
			str = str[len(part):]
			return nil
		})
		assert.Nil(t, err)
		assert.Equal(t, "", str)
	}

	for _, r := range []Rope{
		New("abc").AppendString("def").AppendString("ghi"),
	} {
		str := r.String()
		err := r.Walk(func(part string) error {
			assert.Equal(t, str[:len(part)], part)
			str = str[len(part):]
			if len(str) < 4 {
				return errors.New("stop now")
			}
			return nil
		})
		assert.Equal(t, err, errors.New("stop now"))
		assert.True(t, 0 < len(str) && len(str) < 4)
	}

}

func TestReadAt(t *testing.T) {
	want := treeR.Bytes()

	buf := make([]byte, len(want)+1)

	for start := 0; start < len(buf); start++ {
		for end := start; end <= len(buf); end++ {
			length := end - start
			b := buf[0:length]
			n, err := treeR.ReadAt(b, int64(start))

			// Basic io.ReaderAt contract
			assert.True(t, n <= length)
			if n < length {
				assert.Equal(t, io.EOF, err)
			}

			// Expected actual end and length
			eEnd := end
			if eEnd > len(want) {
				eEnd = len(want)
			}
			eLen := eEnd - start

			// Check for correctness
			assert.Equal(t, eLen, n)
			if eLen < length {
				assert.Equal(t, io.EOF, err)
			} else if eLen > length {
				assert.Nil(t, err)
			} else {
				assert.True(t, err == nil || err == io.EOF)
			}

			assert.Equal(t, want[start:eEnd], b[:n])
		}
	}
}
