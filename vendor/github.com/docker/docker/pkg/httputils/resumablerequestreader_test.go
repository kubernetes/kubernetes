package httputils

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResumableRequestReader(t *testing.T) {

	srvtxt := "some response text data"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, srvtxt)
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}

	client := &http.Client{}
	retries := uint32(5)
	imgSize := int64(len(srvtxt))

	resreq := ResumableRequestReader(client, req, retries, imgSize)
	defer resreq.Close()

	data, err := ioutil.ReadAll(resreq)
	if err != nil {
		t.Fatal(err)
	}

	resstr := strings.TrimSuffix(string(data), "\n")

	if resstr != srvtxt {
		t.Errorf("resstr != srvtxt")
	}
}

func TestResumableRequestReaderWithInitialResponse(t *testing.T) {

	srvtxt := "some response text data"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, srvtxt)
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}

	client := &http.Client{}
	retries := uint32(5)
	imgSize := int64(len(srvtxt))

	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	resreq := ResumableRequestReaderWithInitialResponse(client, req, retries, imgSize, res)
	defer resreq.Close()

	data, err := ioutil.ReadAll(resreq)
	if err != nil {
		t.Fatal(err)
	}

	resstr := strings.TrimSuffix(string(data), "\n")

	if resstr != srvtxt {
		t.Errorf("resstr != srvtxt")
	}
}
