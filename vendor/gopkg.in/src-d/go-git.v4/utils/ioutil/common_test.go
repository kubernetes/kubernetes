package ioutil

import (
	"bytes"
	"context"
	"io/ioutil"
	"strings"
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type CommonSuite struct{}

var _ = Suite(&CommonSuite{})

type closer struct {
	called int
}

func (c *closer) Close() error {
	c.called++
	return nil
}

func (s *CommonSuite) TestNonEmptyReader_Empty(c *C) {
	var buf bytes.Buffer
	r, err := NonEmptyReader(&buf)
	c.Assert(err, Equals, ErrEmptyReader)
	c.Assert(r, IsNil)
}

func (s *CommonSuite) TestNonEmptyReader_NonEmpty(c *C) {
	buf := bytes.NewBuffer([]byte("1"))
	r, err := NonEmptyReader(buf)
	c.Assert(err, IsNil)
	c.Assert(r, NotNil)

	read, err := ioutil.ReadAll(r)
	c.Assert(err, IsNil)
	c.Assert(string(read), Equals, "1")
}

func (s *CommonSuite) TestNewReadCloser(c *C) {
	buf := bytes.NewBuffer([]byte("1"))
	closer := &closer{}
	r := NewReadCloser(buf, closer)

	read, err := ioutil.ReadAll(r)
	c.Assert(err, IsNil)
	c.Assert(string(read), Equals, "1")

	c.Assert(r.Close(), IsNil)
	c.Assert(closer.called, Equals, 1)
}

func (s *CommonSuite) TestNewContextReader(c *C) {
	buf := bytes.NewBuffer([]byte("12"))
	ctx, close := context.WithCancel(context.Background())

	r := NewContextReader(ctx, buf)

	b := make([]byte, 1)
	n, err := r.Read(b)
	c.Assert(n, Equals, 1)
	c.Assert(err, IsNil)

	close()
	n, err = r.Read(b)
	c.Assert(n, Equals, 0)
	c.Assert(err, NotNil)
}

func (s *CommonSuite) TestNewContextReadCloser(c *C) {
	buf := NewReadCloser(bytes.NewBuffer([]byte("12")), &closer{})
	ctx, close := context.WithCancel(context.Background())

	r := NewContextReadCloser(ctx, buf)

	b := make([]byte, 1)
	n, err := r.Read(b)
	c.Assert(n, Equals, 1)
	c.Assert(err, IsNil)

	close()
	n, err = r.Read(b)
	c.Assert(n, Equals, 0)
	c.Assert(err, NotNil)

	c.Assert(r.Close(), IsNil)
}

func (s *CommonSuite) TestNewContextWriter(c *C) {
	buf := bytes.NewBuffer(nil)
	ctx, close := context.WithCancel(context.Background())

	r := NewContextWriter(ctx, buf)

	n, err := r.Write([]byte("1"))
	c.Assert(n, Equals, 1)
	c.Assert(err, IsNil)

	close()
	n, err = r.Write([]byte("1"))
	c.Assert(n, Equals, 0)
	c.Assert(err, NotNil)
}

func (s *CommonSuite) TestNewContextWriteCloser(c *C) {
	buf := NewWriteCloser(bytes.NewBuffer(nil), &closer{})
	ctx, close := context.WithCancel(context.Background())

	w := NewContextWriteCloser(ctx, buf)

	n, err := w.Write([]byte("1"))
	c.Assert(n, Equals, 1)
	c.Assert(err, IsNil)

	close()
	n, err = w.Write([]byte("1"))
	c.Assert(n, Equals, 0)
	c.Assert(err, NotNil)

	c.Assert(w.Close(), IsNil)
}

func (s *CommonSuite) TestNewWriteCloserOnError(c *C) {
	buf := NewWriteCloser(bytes.NewBuffer(nil), &closer{})

	ctx, close := context.WithCancel(context.Background())

	var called error
	w := NewWriteCloserOnError(NewContextWriteCloser(ctx, buf), func(err error) {
		called = err
	})

	close()
	w.Write(nil)

	c.Assert(called, NotNil)
}

func (s *CommonSuite) TestNewReadCloserOnError(c *C) {
	buf := NewReadCloser(bytes.NewBuffer(nil), &closer{})
	ctx, close := context.WithCancel(context.Background())

	var called error
	w := NewReadCloserOnError(NewContextReadCloser(ctx, buf), func(err error) {
		called = err
	})

	close()
	w.Read(nil)

	c.Assert(called, NotNil)
}
func ExampleCheckClose() {
	// CheckClose is commonly used with named return values
	f := func() (err error) {
		// Get a io.ReadCloser
		r := ioutil.NopCloser(strings.NewReader("foo"))

		// defer CheckClose call with an io.Closer and pointer to error
		defer CheckClose(r, &err)

		// ... work with r ...

		// if err is not nil, CheckClose will assign any close errors to it
		return err
	}

	err := f()
	if err != nil {
		panic(err)
	}
}
