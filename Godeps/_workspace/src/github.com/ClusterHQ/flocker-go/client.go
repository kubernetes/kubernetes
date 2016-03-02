package flocker

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// From https://github.com/ClusterHQ/flocker-docker-plugin/blob/master/flockerdockerplugin/adapter.py#L18
const defaultVolumeSize = json.Number("107374182400")

var (
	// A volume can take a long time to be available, if we don't want
	// Kubernetes to wait forever we need to stop trying after some time, that
	// time is defined here
	timeoutWaitingForVolume = 2 * time.Minute
	tickerWaitingForVolume  = 5 * time.Second

	errStateNotFound         = errors.New("State not found by Dataset ID")
	errConfigurationNotFound = errors.New("Configuration not found by Name")

	errFlockerControlServiceHost = errors.New("The volume config must have a key CONTROL_SERVICE_HOST defined in the OtherAttributes field")
	errFlockerControlServicePort = errors.New("The volume config must have a key CONTROL_SERVICE_PORT defined in the OtherAttributes field")

	errVolumeAlreadyExists = errors.New("The volume already exists")
	errVolumeDoesNotExist  = errors.New("The volume does not exist")

	errUpdatingDataset = errors.New("It was impossible to update the dataset")
)

// Clientable exposes the needed methods to implement your own Flocker Client.
type Clientable interface {
	CreateDataset(metaName string) (*DatasetState, error)

	GetDatasetState(datasetID string) (*DatasetState, error)
	GetDatasetID(metaName string) (datasetID string, err error)
	GetPrimaryUUID() (primaryUUID string, err error)

	UpdatePrimaryForDataset(primaryUUID, datasetID string) (*DatasetState, error)
}

// Client is a default Flocker Client.
type Client struct {
	*http.Client

	schema  string
	host    string
	port    int
	version string

	clientIP string

	maximumSize json.Number
}

// NewClient creates a wrapper over http.Client to communicate with the flocker control service.
func NewClient(host string, port int, clientIP string, caCertPath, keyPath, certPath string) (*Client, error) {
	client, err := newTLSClient(caCertPath, keyPath, certPath)
	if err != nil {
		return nil, err
	}

	return &Client{
		Client:      client,
		schema:      "https",
		host:        host,
		port:        port,
		version:     "v1",
		maximumSize: defaultVolumeSize,
		clientIP:    clientIP,
	}, nil
}

/*
request do a request using the http.Client embedded to the control service
and returns the response or an error in case it happens.

Note: you will need to deal with the response body call to Close if you
don't want to deal with problems later.
*/
func (c Client) request(method, url string, payload interface{}) (*http.Response, error) {
	var (
		b   []byte
		err error
	)

	if method == "POST" { // Just allow payload on POST
		b, err = json.Marshal(payload)
		if err != nil {
			return nil, err
		}
	}

	req, err := http.NewRequest(method, url, bytes.NewBuffer(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	// REMEMBER TO CLOSE THE BODY IN THE OUTSIDE FUNCTION
	return c.Do(req)
}

// post performs a post request with the indicated payload
func (c Client) post(url string, payload interface{}) (*http.Response, error) {
	return c.request("POST", url, payload)
}

// get performs a get request
func (c Client) get(url string) (*http.Response, error) {
	return c.request("GET", url, nil)
}

// getURL returns a full URI to the control service
func (c Client) getURL(path string) string {
	return fmt.Sprintf("%s://%s:%d/%s/%s", c.schema, c.host, c.port, c.version, path)
}

type configurationPayload struct {
	Deleted     bool            `json:"deleted"`
	Primary     string          `json:"primary"`
	DatasetID   string          `json:"dataset_id,omitempty"`
	MaximumSize json.Number     `json:"maximum_size,omitempty"`
	Metadata    metadataPayload `json:"metadata,omitempty"`
}

type metadataPayload struct {
	Name string `json:"name,omitempty"`
}

type DatasetState struct {
	Path        string      `json:"path"`
	DatasetID   string      `json:"dataset_id"`
	Primary     string      `json:"primary,omitempty"`
	MaximumSize json.Number `json:"maximum_size,omitempty"`
}

type datasetStatePayload struct {
	*DatasetState
}

type nodeStatePayload struct {
	UUID string `json:"uuid"`
	Host string `json:"host"`
}

// findIDInConfigurationsPayload returns the datasetID if it was found in the
// configurations payload, otherwise it will return an error.
func (c Client) findIDInConfigurationsPayload(body io.ReadCloser, name string) (datasetID string, err error) {
	var configurations []configurationPayload
	if err = json.NewDecoder(body).Decode(&configurations); err == nil {
		for _, r := range configurations {
			if r.Metadata.Name == name {
				return r.DatasetID, nil
			}
		}
		return "", errConfigurationNotFound
	}
	return "", err
}

// GetPrimaryUUID returns the UUID of the primary Flocker Control Service for
// the given host.
func (c Client) GetPrimaryUUID() (uuid string, err error) {
	resp, err := c.get(c.getURL("state/nodes"))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var states []nodeStatePayload
	if err = json.NewDecoder(resp.Body).Decode(&states); err == nil {
		for _, s := range states {
			if s.Host == c.clientIP {
				return s.UUID, nil
			}
		}
		return "", errStateNotFound
	}
	return "", err
}

// GetDatasetState performs a get request to get the state of the given datasetID, if
// something goes wrong or the datasetID was not found it returns an error.
func (c Client) GetDatasetState(datasetID string) (*DatasetState, error) {
	resp, err := c.get(c.getURL("state/datasets"))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var states []datasetStatePayload
	if err = json.NewDecoder(resp.Body).Decode(&states); err == nil {
		for _, s := range states {
			if s.DatasetID == datasetID {
				return s.DatasetState, nil
			}
		}
		return nil, errStateNotFound
	}

	return nil, err
}

/*
CreateDataset creates a volume in Flocker, waits for it to be ready and
returns the dataset id.

This process is a little bit complex but follows this flow:

1. Find the Flocker Control Service UUID
2. Try to create the dataset
3. If it already exists an error is returned
4. If it didn't previously exist, wait for it to be ready
*/
func (c Client) CreateDataset(metaName string) (*DatasetState, error) {
	// 1) Find the primary Flocker UUID
	// Note: it could be cached, but doing this query we health check it
	primary, err := c.GetPrimaryUUID()
	if err != nil {
		return nil, err
	}

	// 2) Try to create the dataset in the given Primary
	payload := configurationPayload{
		Primary:     primary,
		MaximumSize: json.Number(c.maximumSize),
		Metadata: metadataPayload{
			Name: metaName,
		},
	}

	resp, err := c.post(c.getURL("configuration/datasets"), payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// 3) Return if the dataset was previously created
	if resp.StatusCode == http.StatusConflict {
		return nil, errVolumeAlreadyExists
	}

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Expected: {1,2}xx creating the volume, got: %d", resp.StatusCode)
	}

	var p configurationPayload
	if err := json.NewDecoder(resp.Body).Decode(&p); err != nil {
		return nil, err
	}

	// 4) Wait until the dataset is ready for usage. In case it never gets
	// ready there is a timeoutChan that will return an error
	timeoutChan := time.NewTimer(timeoutWaitingForVolume).C
	tickChan := time.NewTicker(tickerWaitingForVolume).C

	for {
		if s, err := c.GetDatasetState(p.DatasetID); err == nil {
			return s, nil
		} else if err != errStateNotFound {
			return nil, err
		}

		select {
		case <-timeoutChan:
			return nil, err
		case <-tickChan:
			break
		}
	}
}

// UpdatePrimaryForDataset will update the Primary for the given dataset
// returning the current DatasetState.
func (c Client) UpdatePrimaryForDataset(newPrimaryUUID, datasetID string) (*DatasetState, error) {
	payload := struct {
		Primary string `json:"primary"`
	}{
		Primary: newPrimaryUUID,
	}

	url := c.getURL(fmt.Sprintf("configuration/datasets/%s", datasetID))
	resp, err := c.post(url, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return nil, errUpdatingDataset
	}

	var s DatasetState
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		return nil, err
	}

	return &s, nil
}

// GetDatasetID will return the DatasetID found for the given metadata name.
func (c Client) GetDatasetID(metaName string) (datasetID string, err error) {
	resp, err := c.get(c.getURL("configuration/datasets"))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var configurations []configurationPayload
	if err = json.NewDecoder(resp.Body).Decode(&configurations); err == nil {
		for _, c := range configurations {
			if c.Metadata.Name == metaName && c.Deleted == false {
				return c.DatasetID, nil
			}
		}
		return "", errConfigurationNotFound
	}
	return "", err
}
