package opaque

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	restclient "k8s.io/client-go/rest"
)

// body for opaque resource request
type patchOperation struct {
	Operation string `json:"op"`
	Path      string `json:"path"`
	Value     string `json:"value,omitempty"`
}

func toRequestBody(patchStruct patchOperation) ([]byte, error) {
	// body is in form of [{"op": "add", "path": "somepath","value": "somevalue"}]
	return json.Marshal(
		[]patchOperation{
			patchStruct,
		},
	)
}

// TODO: check if kubelet is overriding hostname and return it instead
func getNode() (string, error) {
	return os.Hostname()
}

// Escape forward slashes in the resource name per the JSON Pointer spec.
// See https://tools.ietf.org/html/rfc6901#section-3
func escapeResourcePath(resName api.ResourceName) string {
	replacer := strings.NewReplacer(
		"/", "~1",
		"~", "~0",
	)
	return replacer.Replace(string(resName))
}

// genereate OIR path for patch operation
func opaqueResourcePath(name string) string {
	return fmt.Sprintf("/status/capacity/%s", escapeResourcePath(api.OpaqueIntResourceName(name)))
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

func AdvertiseOpaqueResource(name string, value int) error {
	// prepare body for OIR operation
	patch := patchOperation{
		Path:      opaqueResourcePath(name),
		Operation: "add",
		Value:     fmt.Sprintf("%d", value),
	}

	return makeRequest(patch)
}

func RemoveOpaqueResource(name string) error {
	// prepare body for OIR operation
	patch := patchOperation{
		Operation: "remove",
		Path:      opaqueResourcePath(name),
	}

	return makeRequest(patch)
}

func makeRequest(patchBody patchOperation) error {
	body, err := toRequestBody(patchBody)
	if err != nil {
		return fmt.Errorf("Cannot marshall request's body to json: %v", err)
	}

	// get InClusterConfig
	config, err := getClientConfig()
	if err != nil {
		return fmt.Errorf("Cannot get client config: %v", err)
	}

	// create a k8s-client
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("Cannot create k8s client: %v", err)
	}
	node, err := getNode()
	if err != nil {
		return fmt.Errorf("Cannot get node name: %v", err)
	}

	// make patch request to add/remove OIR
	return clientset.CoreV1().RESTClient().Patch(types.JSONPatchType).Resource("nodes").Name(node).SubResource("status").Body(body).Do().Error()
}
