package flocker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/volume"

	"github.com/stretchr/testify/assert"
)

func TestMaximumSizeIs1024Multiple(t *testing.T) {
	assert := assert.New(t)

	n, err := strconv.Atoi(string(defaultVolumeSize))
	assert.NoError(err)
	assert.Equal(0, n%1024)
}

func TestPost(t *testing.T) {
	const (
		expectedPayload    = "foobar"
		expectedStatusCode = 418
	)

	assert := assert.New(t)

	type payload struct {
		Test string `json:"test"`
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var result payload
		err := json.NewDecoder(r.Body).Decode(&result)
		assert.NoError(err)
		assert.Equal(expectedPayload, result.Test)
		w.WriteHeader(expectedStatusCode)
	}))
	defer ts.Close()

	c := Client{Client: &http.Client{}}

	resp, err := c.post(ts.URL, payload{expectedPayload})
	assert.NoError(err)
	assert.Equal(expectedStatusCode, resp.StatusCode)
}

func TestGet(t *testing.T) {
	const (
		expectedStatusCode = 418
	)

	assert := assert.New(t)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(expectedStatusCode)
	}))
	defer ts.Close()

	c := Client{Client: &http.Client{}}

	resp, err := c.get(ts.URL)
	assert.NoError(err)
	assert.Equal(expectedStatusCode, resp.StatusCode)
}

func TestFindIDInConfigurationsPayload(t *testing.T) {
	const (
		searchedName = "search-for-this-name"
		expected     = "The-42-id"
	)
	assert := assert.New(t)

	c := Client{}

	payload := fmt.Sprintf(
		`[{"dataset_id": "1-2-3", "metadata": {"name": "test"}}, {"dataset_id": "The-42-id", "metadata": {"name": "%s"}}]`,
		searchedName,
	)

	id, err := c.findIDInConfigurationsPayload(
		ioutil.NopCloser(bytes.NewBufferString(payload)), searchedName,
	)
	assert.NoError(err)
	assert.Equal(expected, id)

	id, err = c.findIDInConfigurationsPayload(
		ioutil.NopCloser(bytes.NewBufferString(payload)), "it will not be found",
	)
	assert.Equal(errConfigurationNotFound, err)

	id, err = c.findIDInConfigurationsPayload(
		ioutil.NopCloser(bytes.NewBufferString("invalid { json")), "",
	)
	assert.Error(err)
}

func TestFindPrimaryUUID(t *testing.T) {
	const expectedPrimary = "primary-uuid"
	assert := assert.New(t)

	var (
		mockedHost    = "127.0.0.1"
		mockedPrimary = expectedPrimary
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal("GET", r.Method)
		assert.Equal("/v1/state/nodes", r.URL.Path)
		w.Write([]byte(fmt.Sprintf(`[{"host": "%s", "uuid": "%s"}]`, mockedHost, mockedPrimary)))
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)
	assert.NoError(err)

	mockedPrimary = expectedPrimary
	primary, err := c.GetPrimaryUUID()
	assert.NoError(err)
	assert.Equal(expectedPrimary, primary)

	c.clientIP = "not.found"
	_, err = c.GetPrimaryUUID()
	assert.Equal(errStateNotFound, err)
}

func TestGetURL(t *testing.T) {
	const (
		expectedHost = "host"
		expectedPort = 42
	)

	assert := assert.New(t)

	c := newFlockerTestClient(expectedHost, expectedPort)
	var expectedURL = fmt.Sprintf("%s://%s:%d/v1/test", c.schema, expectedHost, expectedPort)

	url := c.getURL("test")
	assert.Equal(expectedURL, url)
}

func getHostAndPortFromTestServer(ts *httptest.Server) (string, int, error) {
	tsURL, err := url.Parse(ts.URL)
	if err != nil {
		return "", 0, err
	}

	hostSplits := strings.Split(tsURL.Host, ":")

	port, err := strconv.Atoi(hostSplits[1])
	if err != nil {
		return "", 0, nil
	}
	return hostSplits[0], port, nil
}

func getVolumeConfig(host string, port int) volume.VolumeConfig {
	return volume.VolumeConfig{
		OtherAttributes: map[string]string{
			"CONTROL_SERVICE_HOST": host,
			"CONTROL_SERVICE_PORT": strconv.Itoa(port),
		},
	}
}

func TestHappyPathCreateDatasetFromNonExistent(t *testing.T) {
	const (
		expectedDatasetName = "dir"
		expectedPrimary     = "A-B-C-D"
		expectedDatasetID   = "datasetID"
	)
	expectedPath := fmt.Sprintf("/flocker/%s", expectedDatasetID)

	assert := assert.New(t)
	var (
		numCalls int
		err      error
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		numCalls++
		switch numCalls {
		case 1:
			assert.Equal("GET", r.Method)
			assert.Equal("/v1/state/nodes", r.URL.Path)
			w.Write([]byte(fmt.Sprintf(`[{"host": "127.0.0.1", "uuid": "%s"}]`, expectedPrimary)))
		case 2:
			assert.Equal("POST", r.Method)
			assert.Equal("/v1/configuration/datasets", r.URL.Path)

			var c configurationPayload
			err := json.NewDecoder(r.Body).Decode(&c)
			assert.NoError(err)
			assert.Equal(expectedPrimary, c.Primary)
			assert.Equal(defaultVolumeSize, c.MaximumSize)
			assert.Equal(expectedDatasetName, c.Metadata.Name)

			w.Write([]byte(fmt.Sprintf(`{"dataset_id": "%s"}`, expectedDatasetID)))
		case 3:
			assert.Equal("GET", r.Method)
			assert.Equal("/v1/state/datasets", r.URL.Path)
			w.Write([]byte(`[]`))
		case 4:
			assert.Equal("GET", r.Method)
			assert.Equal("/v1/state/datasets", r.URL.Path)
			w.Write([]byte(fmt.Sprintf(`[{"dataset_id": "%s", "path": "/flocker/%s"}]`, expectedDatasetID, expectedDatasetID)))
		}
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)
	assert.NoError(err)

	tickerWaitingForVolume = 1 * time.Millisecond // TODO: this is overriding globally

	s, err := c.CreateDataset(expectedDatasetName)
	assert.NoError(err)
	assert.Equal(expectedPath, s.Path)
}

func TestCreateDatasetThatAlreadyExists(t *testing.T) {
	const (
		datasetName     = "dir"
		expectedPrimary = "A-B-C-D"
	)

	assert := assert.New(t)
	var numCalls int

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		numCalls++
		switch numCalls {
		case 1:
			assert.Equal("GET", r.Method)
			assert.Equal("/v1/state/nodes", r.URL.Path)
			w.Write([]byte(fmt.Sprintf(`[{"host": "127.0.0.1", "uuid": "%s"}]`, expectedPrimary)))
		case 2:
			assert.Equal("POST", r.Method)
			assert.Equal("/v1/configuration/datasets", r.URL.Path)
			w.WriteHeader(http.StatusConflict)
		}
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)
	assert.NoError(err)

	_, err = c.CreateDataset(datasetName)
	assert.Equal(errVolumeAlreadyExists, err)
}

func TestUpdatePrimaryForDataset(t *testing.T) {
	const (
		dir               = "dir"
		expectedPrimary   = "the-new-primary"
		expectedDatasetID = "datasetID"
	)
	expectedURL := fmt.Sprintf("/v1/configuration/datasets/%s", expectedDatasetID)

	assert := assert.New(t)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal("POST", r.Method)
		assert.Equal(expectedURL, r.URL.Path)

		var c configurationPayload
		err := json.NewDecoder(r.Body).Decode(&c)
		assert.NoError(err)

		assert.Equal(expectedPrimary, c.Primary)

		w.Write([]byte(fmt.Sprintf(`{"dataset_id": "%s", "path": "just-to-double-check"}`, expectedDatasetID)))
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)
	assert.NoError(err)

	s, err := c.UpdatePrimaryForDataset(expectedPrimary, expectedDatasetID)
	assert.NoError(err)
	assert.Equal(expectedDatasetID, s.DatasetID)
	assert.NotEqual("", s.Path)
}

func TestInterfaceIsImplemented(t *testing.T) {
	assert.Implements(t, (*Clientable)(nil), Client{})
}

func newFlockerTestClient(host string, port int) *Client {
	return &Client{
		Client:      &http.Client{},
		host:        host,
		port:        port,
		version:     "v1",
		schema:      "http",
		maximumSize: defaultVolumeSize,
		clientIP:    "127.0.0.1",
	}
}
