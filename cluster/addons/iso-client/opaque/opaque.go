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

// body for opaque resource request
type patchOperation struct {
	Operation string `json:"op"`
	Path      string `json:"path"`
	Value     string `json:"value"`
}

func toRequestBody(operation string, name string, value string) ([]byte, error) {
	// body is in form of [{"op": "add", "path": "somepath","value": "somevalue"}]
	return json.Marshal([]patchOperation{
		patchOperation{
			Operation: operation,
			Path:      generateOpaqueResourcePath(name),
			Value:     value,
		},
	})
}

// TODO: check if kubelet is overriding hostname and return it instead
func getNode() (string, error) {
	return os.Hostname()
}

// path for opaque resources
func generateOpaqueResourcePath(name string) string {
	return fmt.Sprintf("/status/capacity/pod.alpha.kubernetes.io~1opaque-int-resource-%s", name)
}

// generate url for adding or removing opaque resources
func generateOpaqueResourceUrl() (string, error) {
	host, err := getApiServer()
	if err != nil {
		return host, err
	}
	node, err := getNode()
	if err != nil {
		return node, err
	}
	return fmt.Sprintf("%s/api/v1/nodes/%s/status", host, node), nil
}

// Getting config for accesing apiserver, assuming pod ir run within the cluster
func getClientConfig() (*restclient.Config, error) {
	return restclient.InClusterConfig()
}

func getApiServer() (string, error) {
	config, err := getClientConfig()
	if err != nil {
		return "", err
	}
	return config.Host, nil
}

// prepare PATCH request for adding/removing opaque resources
func prepareRequest(body []byte, url string) (*http.Request, error) {
	return http.NewRequest(http.MethodPatch, url, bytes.NewBuffer(body))
}

//Setting application/json-patch+json header
func setPatchHeader(req *http.Request) {
	req.Header.Set("Content-Type", "application/json-patch+json")
}

func createHttpClient() (*http.Client, error) {
	// TODO: test it against insecure apiserver
	config, err := getClientConfig()
	if err != nil {
		return nil, err
	}
	transportConfig, err := config.TransportConfig()
	if err != nil {
		return nil, err
	}
	transport, err := transport.New(transportConfig)
	if err != nil {
		return nil, err
	}
	return &http.Client{Transport: transport}, nil
}

func AdvertiseOpaqueResource(name string, value string) error {
	return makeRequest("add", name, value)
}

func RemoveOpaqueResource(name string) error {
	return makeRequest("remove", name, "1")
}

func makeRequest(operation string, name string, value string) error {
	body, err := toRequestBody(operation, name, value)
	if err != nil {
		return err
	}

	url, err := generateOpaqueResourceUrl()
	if err != nil {
		return err
	}

	req, err := prepareRequest(body, url)
	if err != nil {
		return err
	}
	// Setting proper header for PATCH request
	setPatchHeader(req)

	client, err := createHttpClient()
	if err != nil {
		return err
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Cannot set  opaque %s", name)
	}

	return nil
}
