package restful

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

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
}
