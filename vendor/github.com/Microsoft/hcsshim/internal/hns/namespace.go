package hns

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"
)

type namespaceRequest struct {
	IsDefault bool `json:",omitempty"`
}

type namespaceEndpointRequest struct {
	ID string `json:"Id"`
}

type NamespaceResource struct {
	Type string
	Data json.RawMessage
}

type namespaceResourceRequest struct {
	Type string
	Data interface{}
}

type Namespace struct {
	ID            string
	IsDefault     bool                `json:",omitempty"`
	ResourceList  []NamespaceResource `json:",omitempty"`
	CompartmentId uint32              `json:",omitempty"`
}

func issueNamespaceRequest(id *string, method, subpath string, request interface{}) (*Namespace, error) {
	var err error
	hnspath := "/namespaces/"
	if id != nil {
		hnspath = path.Join(hnspath, *id)
	}
	if subpath != "" {
		hnspath = path.Join(hnspath, subpath)
	}
	var reqJSON []byte
	if request != nil {
		if reqJSON, err = json.Marshal(request); err != nil {
			return nil, err
		}
	}
	var ns Namespace
	err = hnsCall(method, hnspath, string(reqJSON), &ns)
	if err != nil {
		if strings.Contains(err.Error(), "Element not found.") {
			return nil, os.ErrNotExist
		}
		return nil, fmt.Errorf("%s %s: %s", method, hnspath, err)
	}
	return &ns, err
}

func CreateNamespace() (string, error) {
	req := namespaceRequest{}
	ns, err := issueNamespaceRequest(nil, "POST", "", &req)
	if err != nil {
		return "", err
	}
	return ns.ID, nil
}

func RemoveNamespace(id string) error {
	_, err := issueNamespaceRequest(&id, "DELETE", "", nil)
	return err
}

func GetNamespaceEndpoints(id string) ([]string, error) {
	ns, err := issueNamespaceRequest(&id, "GET", "", nil)
	if err != nil {
		return nil, err
	}
	var endpoints []string
	for _, rsrc := range ns.ResourceList {
		if rsrc.Type == "Endpoint" {
			var endpoint namespaceEndpointRequest
			err = json.Unmarshal(rsrc.Data, &endpoint)
			if err != nil {
				return nil, fmt.Errorf("unmarshal endpoint: %s", err)
			}
			endpoints = append(endpoints, endpoint.ID)
		}
	}
	return endpoints, nil
}

func AddNamespaceEndpoint(id string, endpointID string) error {
	resource := namespaceResourceRequest{
		Type: "Endpoint",
		Data: namespaceEndpointRequest{endpointID},
	}
	_, err := issueNamespaceRequest(&id, "POST", "addresource", &resource)
	return err
}

func RemoveNamespaceEndpoint(id string, endpointID string) error {
	resource := namespaceResourceRequest{
		Type: "Endpoint",
		Data: namespaceEndpointRequest{endpointID},
	}
	_, err := issueNamespaceRequest(&id, "POST", "removeresource", &resource)
	return err
}
