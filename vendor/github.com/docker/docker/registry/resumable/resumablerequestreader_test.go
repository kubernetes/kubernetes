package resumable

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestResumableRequestHeaderSimpleErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, world !")
	}))
	defer ts.Close()

	client := &http.Client{}

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	require.NoError(t, err)

	resreq := &requestReader{}
	_, err = resreq.Read([]byte{})
	assert.EqualError(t, err, "client and request can't be nil")

	resreq = &requestReader{
		client:    client,
		request:   req,
		totalSize: -1,
	}
	_, err = resreq.Read([]byte{})
	assert.EqualError(t, err, "failed to auto detect content length")
}

// Not too much failures, bails out after some wait
func TestResumableRequestHeaderNotTooMuchFailures(t *testing.T) {
	client := &http.Client{}

	var badReq *http.Request
	badReq, err := http.NewRequest("GET", "I'm not an url", nil)
	require.NoError(t, err)

	resreq := &requestReader{
		client:       client,
		request:      badReq,
		failures:     0,
		maxFailures:  2,
		waitDuration: 10 * time.Millisecond,
	}
	read, err := resreq.Read([]byte{})
	require.NoError(t, err)
	assert.Equal(t, 0, read)
}

// Too much failures, returns the error
func TestResumableRequestHeaderTooMuchFailures(t *testing.T) {
	client := &http.Client{}

	var badReq *http.Request
	badReq, err := http.NewRequest("GET", "I'm not an url", nil)
	require.NoError(t, err)

	resreq := &requestReader{
		client:      client,
		request:     badReq,
		failures:    0,
		maxFailures: 1,
	}
	defer resreq.Close()

	expectedError := `Get I%27m%20not%20an%20url: unsupported protocol scheme ""`
	read, err := resreq.Read([]byte{})
	assert.EqualError(t, err, expectedError)
	assert.Equal(t, 0, read)
}

type errorReaderCloser struct{}

func (errorReaderCloser) Close() error { return nil }

func (errorReaderCloser) Read(p []byte) (n int, err error) {
	return 0, fmt.Errorf("An error occurred")
}

// If an unknown error is encountered, return 0, nil and log it
func TestResumableRequestReaderWithReadError(t *testing.T) {
	var req *http.Request
	req, err := http.NewRequest("GET", "", nil)
	require.NoError(t, err)

	client := &http.Client{}

	response := &http.Response{
		Status:        "500 Internal Server",
		StatusCode:    500,
		ContentLength: 0,
		Close:         true,
		Body:          errorReaderCloser{},
	}

	resreq := &requestReader{
		client:          client,
		request:         req,
		currentResponse: response,
		lastRange:       1,
		totalSize:       1,
	}
	defer resreq.Close()

	buf := make([]byte, 1)
	read, err := resreq.Read(buf)
	require.NoError(t, err)

	assert.Equal(t, 0, read)
}

func TestResumableRequestReaderWithEOFWith416Response(t *testing.T) {
	var req *http.Request
	req, err := http.NewRequest("GET", "", nil)
	require.NoError(t, err)

	client := &http.Client{}

	response := &http.Response{
		Status:        "416 Requested Range Not Satisfiable",
		StatusCode:    416,
		ContentLength: 0,
		Close:         true,
		Body:          ioutil.NopCloser(strings.NewReader("")),
	}

	resreq := &requestReader{
		client:          client,
		request:         req,
		currentResponse: response,
		lastRange:       1,
		totalSize:       1,
	}
	defer resreq.Close()

	buf := make([]byte, 1)
	_, err = resreq.Read(buf)
	assert.EqualError(t, err, io.EOF.Error())
}

func TestResumableRequestReaderWithServerDoesntSupportByteRanges(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") == "" {
			t.Fatalf("Expected a Range HTTP header, got nothing")
		}
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	require.NoError(t, err)

	client := &http.Client{}

	resreq := &requestReader{
		client:    client,
		request:   req,
		lastRange: 1,
	}
	defer resreq.Close()

	buf := make([]byte, 2)
	_, err = resreq.Read(buf)
	assert.EqualError(t, err, "the server doesn't support byte ranges")
}

func TestResumableRequestReaderWithZeroTotalSize(t *testing.T) {
	srvtxt := "some response text data"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, srvtxt)
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	require.NoError(t, err)

	client := &http.Client{}
	retries := uint32(5)

	resreq := NewRequestReader(client, req, retries, 0)
	defer resreq.Close()

	data, err := ioutil.ReadAll(resreq)
	require.NoError(t, err)

	resstr := strings.TrimSuffix(string(data), "\n")
	assert.Equal(t, srvtxt, resstr)
}

func TestResumableRequestReader(t *testing.T) {
	srvtxt := "some response text data"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, srvtxt)
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	require.NoError(t, err)

	client := &http.Client{}
	retries := uint32(5)
	imgSize := int64(len(srvtxt))

	resreq := NewRequestReader(client, req, retries, imgSize)
	defer resreq.Close()

	data, err := ioutil.ReadAll(resreq)
	require.NoError(t, err)

	resstr := strings.TrimSuffix(string(data), "\n")
	assert.Equal(t, srvtxt, resstr)
}

func TestResumableRequestReaderWithInitialResponse(t *testing.T) {
	srvtxt := "some response text data"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, srvtxt)
	}))
	defer ts.Close()

	var req *http.Request
	req, err := http.NewRequest("GET", ts.URL, nil)
	require.NoError(t, err)

	client := &http.Client{}
	retries := uint32(5)
	imgSize := int64(len(srvtxt))

	res, err := client.Do(req)
	require.NoError(t, err)

	resreq := NewRequestReaderWithInitialResponse(client, req, retries, imgSize, res)
	defer resreq.Close()

	data, err := ioutil.ReadAll(resreq)
	require.NoError(t, err)

	resstr := strings.TrimSuffix(string(data), "\n")
	assert.Equal(t, srvtxt, resstr)
}
