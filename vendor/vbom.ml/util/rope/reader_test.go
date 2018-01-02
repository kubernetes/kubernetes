package rope

import (
	"bytes"
	"io"
	"testing"

	"github.com/bruth/assert"
)

var readerTests []struct {
	r    node
	want string
}

func init() {
	defer disableCoalesce()()

	readerTests = []struct {
		r    node
		want string
	}{
		{
			r:    leaf("abc"),
			want: "abc",
		},
		{
			r:    conc(leaf("abc"), leaf("def"), 0, 0),
			want: "abcdef",
		},
		{
			r:    conc(conc(leaf("abc"), leaf("123"), 0, 0), leaf("def"), 0, 0),
			want: "abc123def",
		},
	}
}

func TestReader(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	for _, test := range readerTests {
		buf.Reset()
		_, _ = io.Copy(buf, NewReader(Rope{test.r}))
		assert.Equal(t, test.want, buf.String())
	}
}

func TestShortRead(t *testing.T) {
	var result []byte
	var buf [2]byte

	for _, test := range readerTests {
		result = result[:0]

		var (
			r   = NewReader(Rope{test.r})
			n   int
			err error
		)
		for err == nil {
			n, err = r.Read(buf[:])

			if err == nil {
				assert.NotEqual(t, n, 0, "Zero-length Read()")
			}

			result = append(result, buf[:n]...)
		}
		assert.Equal(t, test.want, string(result))
		assert.Equal(t, err, io.EOF, "Non-EOF error: "+err.Error())
	}
}

func TestFullRead(t *testing.T) {
	var buf [9]byte // Big enough for all the test strings. Exactly right size for one of them.

	for _, test := range readerTests {
		r := NewReader(Rope{test.r})
		n, err := r.Read(buf[:])

		assert.Equal(t, n, len(test.want), "partial read")
		assert.Equal(t, err, io.EOF, "no immediate EOF")
		assert.Equal(t, test.want, string(buf[:n]), "wrong data")
	}
}
