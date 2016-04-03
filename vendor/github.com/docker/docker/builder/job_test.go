package builder

import (
	"bytes"
	"io/ioutil"
	"testing"
)

var textPlainDockerfile = "FROM busybox"
var binaryContext = []byte{0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00} //xz magic

func TestInspectEmptyResponse(t *testing.T) {
	ct := "application/octet-stream"
	br := ioutil.NopCloser(bytes.NewReader([]byte("")))
	contentType, bReader, err := inspectResponse(ct, br, 0)
	if err == nil {
		t.Fatalf("Should have generated an error for an empty response")
	}
	if contentType != "application/octet-stream" {
		t.Fatalf("Content type should be 'application/octet-stream' but is %q", contentType)
	}
	body, err := ioutil.ReadAll(bReader)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != 0 {
		t.Fatal("response body should remain empty")
	}
}

func TestInspectResponseBinary(t *testing.T) {
	ct := "application/octet-stream"
	br := ioutil.NopCloser(bytes.NewReader(binaryContext))
	contentType, bReader, err := inspectResponse(ct, br, len(binaryContext))
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "application/octet-stream" {
		t.Fatalf("Content type should be 'application/octet-stream' but is %q", contentType)
	}
	body, err := ioutil.ReadAll(bReader)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != len(binaryContext) {
		t.Fatalf("Wrong response size %d, should be == len(binaryContext)", len(body))
	}
	for i := range body {
		if body[i] != binaryContext[i] {
			t.Fatalf("Corrupted response body at byte index %d", i)
		}
	}
}

func TestResponseUnsupportedContentType(t *testing.T) {
	content := []byte(textPlainDockerfile)
	ct := "application/json"
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bReader, err := inspectResponse(ct, br, len(textPlainDockerfile))

	if err == nil {
		t.Fatal("Should have returned an error on content-type 'application/json'")
	}
	if contentType != ct {
		t.Fatalf("Should not have altered content-type: orig: %s, altered: %s", ct, contentType)
	}
	body, err := ioutil.ReadAll(bReader)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != textPlainDockerfile {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestInspectResponseTextSimple(t *testing.T) {
	content := []byte(textPlainDockerfile)
	ct := "text/plain"
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bReader, err := inspectResponse(ct, br, len(content))
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "text/plain" {
		t.Fatalf("Content type should be 'text/plain' but is %q", contentType)
	}
	body, err := ioutil.ReadAll(bReader)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != textPlainDockerfile {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestInspectResponseEmptyContentType(t *testing.T) {
	content := []byte(textPlainDockerfile)
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bodyReader, err := inspectResponse("", br, len(content))
	if err != nil {
		t.Fatal(err)
	}
	if contentType != "text/plain" {
		t.Fatalf("Content type should be 'text/plain' but is %q", contentType)
	}
	body, err := ioutil.ReadAll(bodyReader)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != textPlainDockerfile {
		t.Fatalf("Corrupted response body %s", body)
	}
}
