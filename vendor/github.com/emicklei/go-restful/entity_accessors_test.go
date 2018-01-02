package restful

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

type keyvalue struct {
	readCalled  bool
	writeCalled bool
}

func (kv *keyvalue) Read(req *Request, v interface{}) error {
	//t := reflect.TypeOf(v)
	//rv := reflect.ValueOf(v)
	kv.readCalled = true
	return nil
}

func (kv *keyvalue) Write(resp *Response, status int, v interface{}) error {
	t := reflect.TypeOf(v)
	rv := reflect.ValueOf(v)
	for ix := 0; ix < t.NumField(); ix++ {
		sf := t.Field(ix)
		io.WriteString(resp, sf.Name)
		io.WriteString(resp, "=")
		io.WriteString(resp, fmt.Sprintf("%v\n", rv.Field(ix).Interface()))
	}
	kv.writeCalled = true
	return nil
}

// go test -v -test.run TestKeyValueEncoding ...restful
func TestKeyValueEncoding(t *testing.T) {
	type Book struct {
		Title         string
		Author        string
		PublishedYear int
	}
	kv := new(keyvalue)
	RegisterEntityAccessor("application/kv", kv)
	b := Book{"Singing for Dummies", "john doe", 2015}

	// Write
	httpWriter := httptest.NewRecorder()
	//								Accept									Produces
	resp := Response{httpWriter, "application/kv,*/*;q=0.8", []string{"application/kv"}, 0, 0, true, nil}
	resp.WriteEntity(b)
	t.Log(string(httpWriter.Body.Bytes()))
	if !kv.writeCalled {
		t.Error("Write never called")
	}

	// Read
	bodyReader := bytes.NewReader(httpWriter.Body.Bytes())
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/kv; charset=UTF-8")
	request := NewRequest(httpRequest)
	var bb Book
	request.ReadEntity(&bb)
	if !kv.readCalled {
		t.Error("Read never called")
	}
}
