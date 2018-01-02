package runtime

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestByteStreamConsumer(t *testing.T) {
	cons := ByteStreamConsumer()
	expected := "the data for the stream to be sent over the wire"
	rdr := bytes.NewBufferString(expected)
	var in bytes.Buffer

	if assert.NoError(t, cons.Consume(rdr, &in)) {
		assert.Equal(t, expected, in.String())
	}
}

func TestByteStreamProducer(t *testing.T) {
	cons := ByteStreamProducer()
	var wrtr bytes.Buffer
	expected := "the data for the stream to be sent over the wire"
	out := bytes.NewBufferString(expected)

	if assert.NoError(t, cons.Produce(&wrtr, out)) {
		assert.Equal(t, expected, wrtr.String())
	}
}
