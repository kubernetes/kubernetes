package opaque

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
)

// struct which keeps all information necessary for advertising opaque resources
type OpaqueIntegerResourceAdvertiser struct {
	Node   string
	Name   string
	Value  string
	Config *restclient.Config
}

// body for opaque resource request
type patchOperation struct {
	Operation string `json:"op"`
	Path      string `json:"path"`
	Value     string `json:"value"`
}

// constructor for OpaqueIntegerResourceAdvertiser
func NewOpaqueIntegerResourceAdvertiser(name string, value string) (*OpaqueIntegerResourceAdvertiser, error) {
	node, err := getNode()
	if err != nil {
		return nil, err
	}
	config, err := getClientConfig()
	if err != nil {
		return nil, err
	}
	return &OpaqueIntegerResourceAdvertiser{
		Node:   node,
		Config: config,
		Name:   name,
		Value:  value,
	}, nil
}

func (opaque *OpaqueIntegerResourceAdvertiser) toRequestBody(operation string) ([]byte, error) {
	// body is in form of [{"op": "add", "path": "somepath","value": "somevalue"}]
	return json.Marshal([]patchOperation{
		patchOperation{
			Operation: operation,
			Path:      opaque.generateOpaqueResourcePath(),
			Value:     opaque.Value,
		},
	})
}

// TODO: check if kubelet is overriding hostname and return it instead
func getNode() (string, error) {
	return os.Hostname()
}

// path for opaque resources
func (opaque *OpaqueIntegerResourceAdvertiser) generateOpaqueResourcePath() string {
	return fmt.Sprintf("/status/capacity/pod.alpha.kubernetes.io~1opaque-int-resource-%s", opaque.Name)
}

// generate url for adding or removing opaque resources
func (opaque *OpaqueIntegerResourceAdvertiser) generateOpaqueResourceUrl() string {
	return fmt.Sprintf("%s/api/v1/nodes/%s/status", opaque.Config.Host, opaque.Node)
}

// Getting config for accesing apiserver, assuming pod ir run within the cluster
func getClientConfig() (*restclient.Config, error) {
	return restclient.InClusterConfig()
}

// prepare PATCH request for adding/removing opaque resources
func prepareRequest(body []byte, url string) (*http.Request, error) {
	return http.NewRequest(http.MethodPatch, url, bytes.NewBuffer(body))
}

//Setting application/json-patch+json header
func setPatchHeader(req *http.Request) {
	req.Header.Set("Content-Type", "application/json-patch+json")
}

func (opaque *OpaqueIntegerResourceAdvertiser) createHttpClient() (*http.Client, error) {
	// TODO: test it against insecure apiserver
	transportConfig, err := opaque.Config.TransportConfig()
	if err != nil {
		return nil, err
	}
	transport, err := transport.New(transportConfig)
	if err != nil {
		return nil, err
	}
	return &http.Client{Transport: transport}, nil
}

func (opaque *OpaqueIntegerResourceAdvertiser) AdvertiseOpaqueResource() error {
	return opaque.makeRequest("add")
}

func (opaque *OpaqueIntegerResourceAdvertiser) RemoveOpaqueResource() error {
	return opaque.makeRequest("remove")
}

func (opaque *OpaqueIntegerResourceAdvertiser) makeRequest(operation string) error {
	body, err := opaque.toRequestBody(operation)
	if err != nil {
		return err
	}

	req, err := prepareRequest(body, opaque.generateOpaqueResourceUrl())
	if err != nil {
		return err
	}
	// Setting proper header for PATCH request
	setPatchHeader(req)

	client, err := opaque.createHttpClient()
	if err != nil {
		return err
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Cannot set  opaque %s", opaque.Name)
	}

	return nil
}
