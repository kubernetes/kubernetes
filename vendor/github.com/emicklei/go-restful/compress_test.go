package restful

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

// go test -v -test.run TestGzip ...restful
func TestGzip(t *testing.T) {
	EnableContentEncoding = true
	httpRequest, _ := http.NewRequest("GET", "/test", nil)
	httpRequest.Header.Set("Accept-Encoding", "gzip,deflate")
	httpWriter := httptest.NewRecorder()
	wanted, encoding := wantsCompressedResponse(httpRequest)
	if !wanted {
		t.Fatal("should accept gzip")
	}
	if encoding != "gzip" {
		t.Fatal("expected gzip")
	}
	c, err := NewCompressingResponseWriter(httpWriter, encoding)
	if err != nil {
		t.Fatal(err.Error())
	}
	c.Write([]byte("Hello World"))
	c.Close()
	if httpWriter.Header().Get("Content-Encoding") != "gzip" {
		t.Fatal("Missing gzip header")
	}
	reader, err := gzip.NewReader(httpWriter.Body)
	if err != nil {
		t.Fatal(err.Error())
	}
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		t.Fatal(err.Error())
	}
	if got, want := string(data), "Hello World"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestDeflate(t *testing.T) {
	EnableContentEncoding = true
	httpRequest, _ := http.NewRequest("GET", "/test", nil)
	httpRequest.Header.Set("Accept-Encoding", "deflate,gzip")
	httpWriter := httptest.NewRecorder()
	wanted, encoding := wantsCompressedResponse(httpRequest)
	if !wanted {
		t.Fatal("should accept deflate")
	}
	if encoding != "deflate" {
		t.Fatal("expected deflate")
	}
	c, err := NewCompressingResponseWriter(httpWriter, encoding)
	if err != nil {
		t.Fatal(err.Error())
	}
	c.Write([]byte("Hello World"))
	c.Close()
	if httpWriter.Header().Get("Content-Encoding") != "deflate" {
		t.Fatal("Missing deflate header")
	}
	reader, err := zlib.NewReader(httpWriter.Body)
	if err != nil {
		t.Fatal(err.Error())
	}
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		t.Fatal(err.Error())
	}
	if got, want := string(data), "Hello World"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestGzipDecompressRequestBody(t *testing.T) {
	b := new(bytes.Buffer)
	w := newGzipWriter()
	w.Reset(b)
	io.WriteString(w, `{"msg":"hi"}`)
	w.Flush()
	w.Close()

	req := new(Request)
	httpRequest, _ := http.NewRequest("GET", "/", bytes.NewReader(b.Bytes()))
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Content-Encoding", "gzip")
	req.Request = httpRequest

	doc := make(map[string]interface{})
	req.ReadEntity(&doc)

	if got, want := doc["msg"], "hi"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestZlibDecompressRequestBody(t *testing.T) {
	b := new(bytes.Buffer)
	w := newZlibWriter()
	w.Reset(b)
	io.WriteString(w, `{"msg":"hi"}`)
	w.Flush()
	w.Close()

	req := new(Request)
	httpRequest, _ := http.NewRequest("GET", "/", bytes.NewReader(b.Bytes()))
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Content-Encoding", "deflate")
	req.Request = httpRequest

	doc := make(map[string]interface{})
	req.ReadEntity(&doc)

	if got, want := doc["msg"], "hi"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}
