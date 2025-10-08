package io_test

import (
	"encoding/binary"
	"io/ioutil"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/gogo/protobuf/test"
	example "github.com/gogo/protobuf/test/example"

	"github.com/gogo/protobuf/io"
)

func BenchmarkUint32DelimWriterMarshaller(b *testing.B) {
	w := io.NewUint32DelimitedWriter(ioutil.Discard, binary.BigEndian)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	msg := example.NewPopulatedA(r, true)

	for i := 0; i < b.N; i++ {
		if err := w.WriteMsg(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUint32DelimWriterFallback(b *testing.B) {
	w := io.NewUint32DelimitedWriter(ioutil.Discard, binary.BigEndian)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	msg := test.NewPopulatedNinOptNative(r, true)

	for i := 0; i < b.N; i++ {
		if err := w.WriteMsg(msg); err != nil {
			b.Fatal(err)
		}
	}
}

// Writing the same size messaged twice should not cause another
// reader buffer allocation
func TestUint32SameSizeNoAlloc(t *testing.T) {
	buf := newBuffer()
	writer := io.NewUint32DelimitedWriter(buf, binary.LittleEndian)
	reader := io.NewUint32DelimitedReader(buf, binary.LittleEndian, 1024*1024)

	err := writer.WriteMsg(&test.NinOptNative{Field15: []byte("numbercatinvention")})
	if err != nil {
		t.Fatal(err)
	}
	err = writer.WriteMsg(&test.NinOptNative{Field15: []byte("fastenselectionsky")})
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	msg := &test.NinOptNative{}
	if err := reader.ReadMsg(msg); err != nil {
		t.Fatal(err)
	}
	firstRead := reflect.ValueOf(reader).Elem().FieldByName("buf").Pointer()
	if err := reader.ReadMsg(msg); err != nil {
		t.Fatal(err)
	}
	secondRead := reflect.ValueOf(reader).Elem().FieldByName("buf").Pointer()

	if firstRead != secondRead {
		t.Fatalf("reader buf byte slice pointer did not stay the same after second same size read (%d != %d).", firstRead, secondRead)
	}
}
