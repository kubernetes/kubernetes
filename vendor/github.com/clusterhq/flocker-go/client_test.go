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

func mockGetStateNodes(assert *assert.Assertions, data []byte) *Client {

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal("GET", r.Method)
		assert.Equal("/v1/state/nodes", r.URL.Path)
		w.Write(data)
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)

	return c
}

func TestListNodes(t *testing.T) {
	assert := assert.New(t)

	var (
		mockedHost1    = "127.0.0.1"
		mockedPrimary1 = "uuid1"
		mockedHost2    = "127.0.0.2"
		mockedPrimary2 = "uuid2"
	)

	c := mockGetStateNodes(
		assert,
		[]byte(fmt.Sprintf(
			`[{"host": "%s", "uuid": "%s"},{"host": "%s", "uuid": "%s"}]`,
			mockedHost1,
			mockedPrimary1,
			mockedHost2,
			mockedPrimary2,
		)),
	)

	nodes, err := c.ListNodes()
	assert.NoError(err)
	assert.Equal(2, len(nodes))
	assert.Equal(mockedHost1, nodes[0].Host)
	assert.Equal(mockedPrimary1, nodes[0].UUID)
	assert.Equal(mockedHost2, nodes[1].Host)
	assert.Equal(mockedPrimary2, nodes[1].UUID)

	c = mockGetStateNodes(
		assert,
		[]byte(`[]`),
	)

	nodes, err = c.ListNodes()
	assert.NoError(err)
	assert.Equal(0, len(nodes))

}

func TestFindPrimaryUUID(t *testing.T) {
	const expectedPrimary = "primary-uuid"
	assert := assert.New(t)

	var (
		mockedHost    = "127.0.0.1"
		mockedPrimary = expectedPrimary
	)

	c := mockGetStateNodes(
		assert,
		[]byte(fmt.Sprintf(`[{"host": "%s", "uuid": "%s"}]`, mockedHost, mockedPrimary)),
	)

	mockedPrimary = expectedPrimary
	primary, err := c.GetPrimaryUUID()
	assert.NoError(err)
	assert.Equal(expectedPrimary, primary)

	c.clientIP = "not.found"
	_, err = c.GetPrimaryUUID()
	assert.Error(err, "An error was expected")
	assert.True(strings.Contains(err.Error(), "No node found"), "returns right error")
	assert.True(strings.Contains(err.Error(), "not.found"), "returns used client IP")
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

func TestDeleteDatasetExisting(t *testing.T) {
	assert := assert.New(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal("DELETE", r.Method)
		assert.Equal("/v1/configuration/datasets/uuid1", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		b, err := json.Marshal(configurationPayload{
			DatasetID: "uuid1",
			Primary:   "primary1",
			Deleted:   true,
		})
		assert.NoError(err)
		w.Write(b)
	},
	))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)

	err = c.DeleteDataset("uuid1")
	assert.NoError(err)
}

func TestDeleteDatasetNotExisting(t *testing.T) {
	assert := assert.New(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal("DELETE", r.Method)
		assert.Equal("/v1/configuration/datasets/uuid2", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		w.Write([]byte(`{"description": "Dataset not found."}`))
	},
	))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)

	err = c.DeleteDataset("uuid2")
	assert.Error(err)
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

	tickerWaitingForVolume = 1 * time.Millisecond // TODO: this is overriding globally

	s, err := c.CreateDataset(&CreateDatasetOptions{
		Metadata: map[string]string{
			"name": expectedDatasetName,
		},
	})

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

	_, err = c.CreateDataset(&CreateDatasetOptions{
		Metadata: map[string]string{
			"name": datasetName,
		},
	})
	assert.Equal(errVolumeAlreadyExists, err)
}

func TestCreateDatasetThatTimesoutServerSide(t *testing.T) {
	const (
		datasetName       = "dir"
		expectedPrimary   = "A-B-C-D"
		expectedDatasetID = "uuid-1"
	)

	// reduce timeouts for testing
	tickerWaitingForVolume = 1 * time.Microsecond
	timeoutWaitingForVolume = 1 * time.Millisecond

	assert := assert.New(t)
	var numCalls int
	var deleted bool

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
			w.Write([]byte(fmt.Sprintf(`{"dataset_id": "%s"}`, expectedDatasetID)))
		default:
			if r.Method == "GET" {
				assert.Equal("/v1/state/datasets", r.URL.Path)
				w.Write([]byte(`[]`))
			} else if r.Method == "DELETE" {
				assert.Equal("/v1/configuration/datasets/uuid-1", r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				b, err := json.Marshal(configurationPayload{
					DatasetID: expectedDatasetID,
					Primary:   expectedPrimary,
					Deleted:   true,
				})
				assert.NoError(err)
				w.Write(b)
				deleted = true
			} else {
				t.Errorf("Received unexpected call '%s' to '%s'", r.Method, r.URL.Path)
			}
		}
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)

	_, err = c.CreateDataset(&CreateDatasetOptions{
		Metadata: map[string]string{
			"name":    datasetName,
			"primary": expectedPrimary,
		},
	})

	assert.True(numCalls > 3, fmt.Sprintf("Not enough retries getting dataset state: %d", numCalls))
	assert.True(deleted, "Failed dataset was not cleaned up afterwards")
	assert.Equal("Flocker API timeout during dataset creation (datasetID uuid-1): State not found by Dataset ID", err.Error())
}

func TestCreateDatasetThatTimesoutServerSideFailedDelete(t *testing.T) {
	const (
		datasetName       = "dir"
		expectedPrimary   = "A-B-C-D"
		expectedDatasetID = "uuid-1"
	)

	// reduce timeouts for testing
	tickerWaitingForVolume = 1 * time.Microsecond
	timeoutWaitingForVolume = 1 * time.Millisecond

	assert := assert.New(t)
	var numCalls int
	var deleted bool

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
			w.Write([]byte(fmt.Sprintf(`{"dataset_id": "%s"}`, expectedDatasetID)))
		default:
			if r.Method == "GET" {
				assert.Equal("/v1/state/datasets", r.URL.Path)
				w.Write([]byte(`[]`))
			} else if r.Method == "DELETE" {
				assert.Equal("/v1/configuration/datasets/uuid-1", r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(500)
				w.Write([]byte(`unexpected error`))
				deleted = true
			} else {
				t.Errorf("Received unexpected call '%s' to '%s'", r.Method, r.URL.Path)
			}
		}
	}))

	host, port, err := getHostAndPortFromTestServer(ts)
	assert.NoError(err)

	c := newFlockerTestClient(host, port)

	_, err = c.CreateDataset(&CreateDatasetOptions{
		Metadata: map[string]string{
			"name":    datasetName,
			"primary": expectedPrimary,
		},
	})

	assert.True(numCalls > 3, fmt.Sprintf("Not enough retries getting dataset state: %d", numCalls))
	assert.True(deleted, "Failed dataset was not cleaned up afterwards")
	assert.Equal("Flocker API timeout during dataset creation (datasetID uuid-1): State not found by Dataset ID, deletion of dataset failed with Expected: {1,2}xx deleting the dataset uuid-1, got: 500", err.Error())
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

	s, err := c.UpdatePrimaryForDataset(expectedPrimary, expectedDatasetID)
	assert.NoError(err)
	assert.Equal(expectedDatasetID, s.DatasetID)
	assert.NotEqual("", s.Path)
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
