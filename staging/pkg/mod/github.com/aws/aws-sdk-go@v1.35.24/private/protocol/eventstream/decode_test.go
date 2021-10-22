package eventstream

import (
	"bytes"
	"encoding/hex"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
)

func TestWriteEncodedFromDecoded(t *testing.T) {
	cases, err := readPositiveTests("testdata")
	if err != nil {
		t.Fatalf("failed to load positive tests, %v", err)
	}

	for _, c := range cases {
		f, err := ioutil.TempFile(os.TempDir(), "encoded_positive_"+c.Name)
		if err != nil {
			t.Fatalf("failed to open %q, %v", c.Name, err)
		}

		encoder := NewEncoder(f)

		msg := c.Decoded.Message()
		if err := encoder.Encode(msg); err != nil {
			t.Errorf("failed to encode %q, %v", c.Name, err)
		}

		if err = f.Close(); err != nil {
			t.Errorf("expected %v, got %v", "no error", err)
		}
		if err = os.Remove(f.Name()); err != nil {
			t.Errorf("expected %v, got %v", "no error", err)
		}
	}
}

func TestDecoder_Decode(t *testing.T) {
	cases, err := readPositiveTests("testdata")
	if err != nil {
		t.Fatalf("failed to load positive tests, %v", err)
	}

	for _, c := range cases {
		decoder := NewDecoder(bytes.NewBuffer(c.Encoded))

		msg, err := decoder.Decode(nil)
		if err != nil {
			t.Fatalf("%s, expect no decode error, got %v", c.Name, err)
		}

		raw, err := msg.rawMessage() // rawMessage will fail if payload read CRC fails
		if err != nil {
			t.Fatalf("%s, failed to get raw decoded message %v", c.Name, err)
		}

		if e, a := c.Decoded.Length, raw.Length; e != a {
			t.Errorf("%s, expect %v length, got %v", c.Name, e, a)
		}
		if e, a := c.Decoded.HeadersLen, raw.HeadersLen; e != a {
			t.Errorf("%s, expect %v HeadersLen, got %v", c.Name, e, a)
		}
		if e, a := c.Decoded.PreludeCRC, raw.PreludeCRC; e != a {
			t.Errorf("%s, expect %v PreludeCRC, got %v", c.Name, e, a)
		}
		if e, a := Headers(c.Decoded.Headers), msg.Headers; !reflect.DeepEqual(e, a) {
			t.Errorf("%s, expect %v headers, got %v", c.Name, e, a)
		}
		if e, a := c.Decoded.Payload, raw.Payload; !bytes.Equal(e, a) {
			t.Errorf("%s, expect %v payload, got %v", c.Name, e, a)
		}
		if e, a := c.Decoded.CRC, raw.CRC; e != a {
			t.Errorf("%s, expect %v CRC, got %v", c.Name, e, a)
		}
	}
}

func TestDecoder_Decode_Negative(t *testing.T) {
	cases, err := readNegativeTests("testdata")
	if err != nil {
		t.Fatalf("failed to load negative tests, %v", err)
	}

	for _, c := range cases {
		decoder := NewDecoder(bytes.NewBuffer(c.Encoded))

		msg, err := decoder.Decode(nil)
		if err == nil {
			rawMsg, rawMsgErr := msg.rawMessage()
			t.Fatalf("%s, expect error, got none, %s,\n%s\n%#v, %v\n", c.Name,
				c.Err, hex.Dump(c.Encoded), rawMsg, rawMsgErr)
		}
	}
}

var testEncodedMsg = []byte{0, 0, 0, 61, 0, 0, 0, 32, 7, 253, 131, 150, 12, 99, 111, 110, 116, 101, 110, 116, 45, 116, 121, 112, 101, 7, 0, 16, 97, 112, 112, 108, 105, 99, 97, 116, 105, 111, 110, 47, 106, 115, 111, 110, 123, 39, 102, 111, 111, 39, 58, 39, 98, 97, 114, 39, 125, 141, 156, 8, 177}

func TestDecoder_DecodeMultipleMessages(t *testing.T) {
	const (
		expectMsgCount   = 10
		expectPayloadLen = 13
	)

	r := bytes.NewBuffer(nil)
	for i := 0; i < expectMsgCount; i++ {
		r.Write(testEncodedMsg)
	}

	decoder := NewDecoder(r)

	var err error
	var msg Message
	var count int
	for {
		msg, err = decoder.Decode(nil)
		if err != nil {
			break
		}
		count++

		if e, a := expectPayloadLen, len(msg.Payload); e != a {
			t.Errorf("expect %v payload len, got %v", e, a)
		}

		if e, a := []byte(`{'foo':'bar'}`), msg.Payload; !bytes.Equal(e, a) {
			t.Errorf("expect %v payload, got %v", e, a)
		}
	}

	type causer interface {
		Cause() error
	}
	if err != nil && count != expectMsgCount {
		t.Fatalf("expect, no error, got %v", err)
	}

	if e, a := expectMsgCount, count; e != a {
		t.Errorf("expect %v messages read, got %v", e, a)
	}
}

func BenchmarkDecode(b *testing.B) {
	r := bytes.NewReader(testEncodedMsg)
	decoder := NewDecoder(r)
	payloadBuf := make([]byte, 0, 5*1024)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		msg, err := decoder.Decode(payloadBuf)
		if err != nil {
			b.Fatal(err)
		}

		// Release the payload buffer
		payloadBuf = msg.Payload[0:0]
		r.Seek(0, 0)
	}
}

func BenchmarkDecode_NoPayloadBuf(b *testing.B) {
	r := bytes.NewReader(testEncodedMsg)
	decoder := NewDecoder(r)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := decoder.Decode(nil)
		if err != nil {
			b.Fatal(err)
		}
		r.Seek(0, 0)
	}
}
