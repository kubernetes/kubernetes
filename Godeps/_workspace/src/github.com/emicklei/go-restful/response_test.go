package restful

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestWriteHeader(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
	resp.WriteHeader(123)
	if resp.StatusCode() != 123 {
		t.Errorf("Unexpected status code:%d", resp.StatusCode())
	}
}

func TestNoWriteHeader(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
	if resp.StatusCode() != http.StatusOK {
		t.Errorf("Unexpected status code:%d", resp.StatusCode())
	}
}

type food struct {
	Kind string
}

// go test -v -test.run TestMeasureContentLengthXml ...restful
func TestMeasureContentLengthXml(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
	resp.WriteAsXml(food{"apple"})
	if resp.ContentLength() != 76 {
		t.Errorf("Incorrect measured length:%d", resp.ContentLength())
	}
}

// go test -v -test.run TestMeasureContentLengthJson ...restful
func TestMeasureContentLengthJson(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
	resp.WriteAsJson(food{"apple"})
	if resp.ContentLength() != 22 {
		t.Errorf("Incorrect measured length:%d", resp.ContentLength())
	}
}

// go test -v -test.run TestMeasureContentLengthJsonNotPretty ...restful
func TestMeasureContentLengthJsonNotPretty(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, false}
	resp.WriteAsJson(food{"apple"})
	if resp.ContentLength() != 16 {
		t.Errorf("Incorrect measured length:%d", resp.ContentLength())
	}
}

// go test -v -test.run TestMeasureContentLengthWriteErrorString ...restful
func TestMeasureContentLengthWriteErrorString(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
	resp.WriteErrorString(404, "Invalid")
	if resp.ContentLength() != len("Invalid") {
		t.Errorf("Incorrect measured length:%d", resp.ContentLength())
	}
}

// go test -v -test.run TestStatusIsPassedToResponse ...restful
func TestStatusIsPassedToResponse(t *testing.T) {
	for _, each := range []struct {
		write, read int
	}{
		{write: 204, read: 204},
		{write: 304, read: 304},
		{write: 200, read: 200},
		{write: 400, read: 200},
	} {
		httpWriter := httptest.NewRecorder()
		resp := Response{httpWriter, "*/*", []string{"*/*"}, 0, 0, true}
		resp.WriteHeader(each.write)
		if got, want := httpWriter.Code, each.read; got != want {
			t.Error("got %v want %v", got, want)
		}
	}
}

// go test -v -test.run TestStatusCreatedAndContentTypeJson_Issue54 ...restful
func TestStatusCreatedAndContentTypeJson_Issue54(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "application/json", []string{"application/json"}, 0, 0, true}
	resp.WriteHeader(201)
	resp.WriteAsJson(food{"Juicy"})
	if httpWriter.HeaderMap.Get("Content-Type") != "application/json" {
		t.Errorf("Expected content type json but got:%d", httpWriter.HeaderMap.Get("Content-Type"))
	}
	if httpWriter.Code != 201 {
		t.Errorf("Expected status 201 but got:%d", httpWriter.Code)
	}
}

type errorOnWriteRecorder struct {
	*httptest.ResponseRecorder
}

func (e errorOnWriteRecorder) Write(bytes []byte) (int, error) {
	return 0, errors.New("fail")
}

// go test -v -test.run TestLastWriteErrorCaught ...restful
func TestLastWriteErrorCaught(t *testing.T) {
	httpWriter := errorOnWriteRecorder{httptest.NewRecorder()}
	resp := Response{httpWriter, "application/json", []string{"application/json"}, 0, 0, true}
	err := resp.WriteAsJson(food{"Juicy"})
	if err.Error() != "fail" {
		t.Errorf("Unexpected error message:%v", err)
	}
}

// go test -v -test.run TestAcceptStarStar_Issue83 ...restful
func TestAcceptStarStar_Issue83(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	//								Accept									Produces
	resp := Response{httpWriter, "application/bogus,*/*;q=0.8", []string{"application/json"}, 0, 0, true}
	resp.WriteEntity(food{"Juicy"})
	ct := httpWriter.Header().Get("Content-Type")
	if "application/json" != ct {
		t.Errorf("Unexpected content type:%s", ct)
	}
}

// go test -v -test.run TestAcceptSkipStarStar_Issue83 ...restful
func TestAcceptSkipStarStar_Issue83(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	//								Accept									Produces
	resp := Response{httpWriter, " application/xml ,*/* ; q=0.8", []string{"application/json", "application/xml"}, 0, 0, true}
	resp.WriteEntity(food{"Juicy"})
	ct := httpWriter.Header().Get("Content-Type")
	if "application/xml" != ct {
		t.Errorf("Unexpected content type:%s", ct)
	}
}

// go test -v -test.run TestAcceptXmlBeforeStarStar_Issue83 ...restful
func TestAcceptXmlBeforeStarStar_Issue83(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	//								Accept									Produces
	resp := Response{httpWriter, "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", []string{"application/json"}, 0, 0, true}
	resp.WriteEntity(food{"Juicy"})
	ct := httpWriter.Header().Get("Content-Type")
	if "application/json" != ct {
		t.Errorf("Unexpected content type:%s", ct)
	}
}

// go test -v -test.run TestWriteHeaderNoContent_Issue124 ...restful
func TestWriteHeaderNoContent_Issue124(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "text/plain", []string{"text/plain"}, 0, 0, true}
	resp.WriteHeader(http.StatusNoContent)
	if httpWriter.Code != http.StatusNoContent {
		t.Errorf("got %d want %d", httpWriter.Code, http.StatusNoContent)
	}
}

// go test -v -test.run TestStatusCreatedAndContentTypeJson_Issue163 ...restful
func TestStatusCreatedAndContentTypeJson_Issue163(t *testing.T) {
	httpWriter := httptest.NewRecorder()
	resp := Response{httpWriter, "application/json", []string{"application/json"}, 0, 0, true}
	resp.WriteHeader(http.StatusNotModified)
	if httpWriter.Code != http.StatusNotModified {
		t.Errorf("Got %d want %d", httpWriter.Code, http.StatusNotModified)
	}
}
