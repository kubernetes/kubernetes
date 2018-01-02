package remotecontext

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/docker/docker/builder"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var binaryContext = []byte{0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00} //xz magic

func TestSelectAcceptableMIME(t *testing.T) {
	validMimeStrings := []string{
		"application/x-bzip2",
		"application/bzip2",
		"application/gzip",
		"application/x-gzip",
		"application/x-xz",
		"application/xz",
		"application/tar",
		"application/x-tar",
		"application/octet-stream",
		"text/plain",
	}

	invalidMimeStrings := []string{
		"",
		"application/octet",
		"application/json",
	}

	for _, m := range invalidMimeStrings {
		if len(selectAcceptableMIME(m)) > 0 {
			t.Fatalf("Should not have accepted %q", m)
		}
	}

	for _, m := range validMimeStrings {
		if str := selectAcceptableMIME(m); str == "" {
			t.Fatalf("Should have accepted %q", m)
		}
	}
}

func TestInspectEmptyResponse(t *testing.T) {
	ct := "application/octet-stream"
	br := ioutil.NopCloser(bytes.NewReader([]byte("")))
	contentType, bReader, err := inspectResponse(ct, br, 0)
	if err == nil {
		t.Fatal("Should have generated an error for an empty response")
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
	contentType, bReader, err := inspectResponse(ct, br, int64(len(binaryContext)))
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
	content := []byte(dockerfileContents)
	ct := "application/json"
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bReader, err := inspectResponse(ct, br, int64(len(dockerfileContents)))

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
	if string(body) != dockerfileContents {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestInspectResponseTextSimple(t *testing.T) {
	content := []byte(dockerfileContents)
	ct := "text/plain"
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bReader, err := inspectResponse(ct, br, int64(len(content)))
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
	if string(body) != dockerfileContents {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestInspectResponseEmptyContentType(t *testing.T) {
	content := []byte(dockerfileContents)
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bodyReader, err := inspectResponse("", br, int64(len(content)))
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
	if string(body) != dockerfileContents {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestUnknownContentLength(t *testing.T) {
	content := []byte(dockerfileContents)
	ct := "text/plain"
	br := ioutil.NopCloser(bytes.NewReader(content))
	contentType, bReader, err := inspectResponse(ct, br, -1)
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
	if string(body) != dockerfileContents {
		t.Fatalf("Corrupted response body %s", body)
	}
}

func TestMakeRemoteContext(t *testing.T) {
	contextDir, cleanup := createTestTempDir(t, "", "builder-tarsum-test")
	defer cleanup()

	createTestTempFile(t, contextDir, builder.DefaultDockerfileName, dockerfileContents, 0777)

	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	serverURL, _ := url.Parse(server.URL)

	serverURL.Path = "/" + builder.DefaultDockerfileName
	remoteURL := serverURL.String()

	mux.Handle("/", http.FileServer(http.Dir(contextDir)))

	remoteContext, err := MakeRemoteContext(remoteURL, map[string]func(io.ReadCloser) (io.ReadCloser, error){
		mimeTypes.TextPlain: func(rc io.ReadCloser) (io.ReadCloser, error) {
			dockerfile, err := ioutil.ReadAll(rc)
			if err != nil {
				return nil, err
			}

			r, err := archive.Generate(builder.DefaultDockerfileName, string(dockerfile))
			if err != nil {
				return nil, err
			}
			return ioutil.NopCloser(r), nil
		},
	})

	if err != nil {
		t.Fatalf("Error when executing DetectContextFromRemoteURL: %s", err)
	}

	if remoteContext == nil {
		t.Fatal("Remote context should not be nil")
	}

	h, err := remoteContext.Hash(builder.DefaultDockerfileName)
	if err != nil {
		t.Fatalf("failed to compute hash %s", err)
	}

	if expected, actual := "7b6b6b66bee9e2102fbdc2228be6c980a2a23adf371962a37286a49f7de0f7cc", h; expected != actual {
		t.Fatalf("There should be file named %s %s in fileInfoSums", expected, actual)
	}
}

func TestGetWithStatusError(t *testing.T) {
	var testcases = []struct {
		err          error
		statusCode   int
		expectedErr  string
		expectedBody string
	}{
		{
			statusCode:   200,
			expectedBody: "THE BODY",
		},
		{
			statusCode:   400,
			expectedErr:  "with status 400 Bad Request: broke",
			expectedBody: "broke",
		},
	}
	for _, testcase := range testcases {
		ts := httptest.NewServer(
			http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				buffer := bytes.NewBufferString(testcase.expectedBody)
				w.WriteHeader(testcase.statusCode)
				w.Write(buffer.Bytes())
			}),
		)
		defer ts.Close()
		response, err := GetWithStatusError(ts.URL)

		if testcase.expectedErr == "" {
			require.NoError(t, err)

			body, err := testutil.ReadBody(response.Body)
			require.NoError(t, err)
			assert.Contains(t, string(body), testcase.expectedBody)
		} else {
			testutil.ErrorContains(t, err, testcase.expectedErr)
		}
	}
}
