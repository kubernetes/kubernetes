package utilities

import (
	"bytes"
	"io"
	"io/ioutil"
)

// IOReaderFactory takes in an io.Reader and returns a function that will allow you to create a new reader that begins
// at the start of the stream
func IOReaderFactory(r io.Reader) (func() io.Reader, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	return func() io.Reader {
		return bytes.NewReader(b)
	}, nil
}
