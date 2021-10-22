// +build example

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// client.go is an example of a client that will request URLs from a service that
// the client will use to upload and download content with.
//
// The server must be started before the client is run.
//
// Use "--help" command line argument flag to see all options and defaults. If
// filename is not provided the client will read from stdin for uploads and
// write to stdout for downloads.
//
// Usage:
//    go run -tags example client.go -get myObjectKey -f filename
func main() {
	method, filename, key, serverURL := loadConfig()

	var err error

	switch method {
	case GetMethod:
		// Requests the URL from the server that the client will use to download
		// the content from. The content will be written to the file pointed to
		// by filename. Creating it if the file does not exist. If filename is
		// not set the contents will be written to stdout.
		err = downloadFile(serverURL, key, filename)
	case PutMethod:
		// Requests the URL from the service that the client will use to upload
		// content to. The content will be read from the file pointed to by the
		// filename. If the filename is not set, content will be read from stdin.
		err = uploadFile(serverURL, key, filename)
	}

	if err != nil {
		exitError(err)
	}
}

// loadConfig configures the client based on the command line arguments used.
func loadConfig() (method Method, serverURL, key, filename string) {
	var getKey, putKey string
	flag.StringVar(&getKey, "get", "",
		"Downloads the object from S3 by the `key`. Writes the object to a file the filename is provided, otherwise writes to stdout.")
	flag.StringVar(&putKey, "put", "",
		"Uploads data to S3 at the `key` provided. Uploads the file if filename is provided, otherwise reads from stdin.")
	flag.StringVar(&serverURL, "s", "http://127.0.0.1:8080", "Required `URL` the client will request presigned S3 operation from.")
	flag.StringVar(&filename, "f", "", "The `filename` of the file to upload and get from S3.")
	flag.Parse()

	var errs Errors

	if len(serverURL) == 0 {
		errs = append(errs, fmt.Errorf("server URL required"))
	}

	if !((len(getKey) != 0) != (len(putKey) != 0)) {
		errs = append(errs, fmt.Errorf("either `get` or `put` can be provided, and one of the two is required."))
	}

	if len(getKey) > 0 {
		method = GetMethod
		key = getKey
	} else {
		method = PutMethod
		key = putKey
	}

	if len(errs) > 0 {
		fmt.Fprintf(os.Stderr, "Failed to load configuration:%v\n", errs)
		flag.PrintDefaults()
		os.Exit(1)
	}

	return method, filename, key, serverURL
}

// downloadFile will request a URL from the server that the client can download
// the content pointed to by "key". The content will be written to the file
// pointed to by filename, creating the file if it doesn't exist. If filename
// is not set the content will be written to stdout.
func downloadFile(serverURL, key, filename string) error {
	var w *os.File
	if len(filename) > 0 {
		f, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create download file %s, %v", filename, err)
		}
		w = f
	} else {
		w = os.Stdout
	}
	defer w.Close()

	// Get the presigned URL from the remote service.
	req, err := getPresignedRequest(serverURL, "GET", key, 0)
	if err != nil {
		return fmt.Errorf("failed to get get presigned request, %v", err)
	}

	// Gets the file contents with the URL provided by the service.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do GET request, %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to get S3 object, %d:%s",
			resp.StatusCode, resp.Status)
	}

	if _, err = io.Copy(w, resp.Body); err != nil {
		return fmt.Errorf("failed to write S3 object, %v", err)
	}

	return nil
}

// uploadFile will request a URL from the service that the client can use to
// upload content to. The content will be read from the file pointed to by filename.
// If filename is not set the content will be read from stdin.
func uploadFile(serverURL, key, filename string) error {
	var r io.ReadCloser
	var size int64
	if len(filename) > 0 {
		f, err := os.Open(filename)
		if err != nil {
			return fmt.Errorf("failed to open upload file %s, %v", filename, err)
		}

		// Get the size of the file so that the constraint of Content-Length
		// can be included with the presigned URL. This can be used by the
		// server or client to ensure the content uploaded is of a certain size.
		//
		// These constraints can further be expanded to include things like
		// Content-Type. Additionally constraints such as X-Amz-Content-Sha256
		// header set restricting the content of the file to only the content
		// the client initially made the request with. This prevents the object
		// from being overwritten or used to upload other unintended content.
		stat, err := f.Stat()
		if err != nil {
			return fmt.Errorf("failed to stat file, %s, %v", filename, err)
		}

		size = stat.Size()
		r = f
	} else {
		buf := &bytes.Buffer{}
		io.Copy(buf, os.Stdin)
		size = int64(buf.Len())

		r = ioutil.NopCloser(buf)
	}
	defer r.Close()

	// Get the Presigned URL from the remote service. Pass in the file's
	// size if it is known so that the presigned URL returned will be required
	// to be used with the size of content requested.
	req, err := getPresignedRequest(serverURL, "PUT", key, size)
	if err != nil {
		return fmt.Errorf("failed to get put presigned request, %v", err)
	}
	req.Body = r

	// Upload the file contents to S3.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do PUT request, %v", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to put S3 object, %d:%s",
			resp.StatusCode, resp.Status)
	}

	return nil
}

// getPresignRequest will request a URL from the service for the content specified
// by the key and method. Returns a constructed Request that can be used to
// upload or download content with based on the method used.
//
// If the PUT method is used the request's Body will need to be set on the returned
// request value.
func getPresignedRequest(serverURL, method, key string, contentLen int64) (*http.Request, error) {
	u := fmt.Sprintf("%s/presign/%s?method=%s&contentLength=%d",
		serverURL, key, method, contentLen,
	)

	resp, err := http.Get(u)
	if err != nil {
		return nil, fmt.Errorf("failed to make request for presigned URL, %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get valid presign response, %s", resp.Status)
	}

	p := PresignResp{}
	if err := json.NewDecoder(resp.Body).Decode(&p); err != nil {
		return nil, fmt.Errorf("failed to decode response body, %v", err)
	}

	req, err := http.NewRequest(p.Method, p.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build presigned request, %v", err)
	}

	for k, vs := range p.Header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	// Need to ensure that the content length member is set of the HTTP Request
	// or the request will not be transmitted correctly with a content length
	// value across the wire.
	if contLen := req.Header.Get("Content-Length"); len(contLen) > 0 {
		req.ContentLength, _ = strconv.ParseInt(contLen, 10, 64)
	}

	return req, nil
}

type Method int

const (
	PutMethod Method = iota
	GetMethod
)

type Errors []error

func (es Errors) Error() string {
	out := make([]string, len(es))
	for _, e := range es {
		out = append(out, e.Error())
	}
	return strings.Join(out, "\n")
}

type PresignResp struct {
	Method, URL string
	Header      http.Header
}

func exitError(err error) {
	fmt.Fprintln(os.Stderr, err.Error())
	os.Exit(1)
}
