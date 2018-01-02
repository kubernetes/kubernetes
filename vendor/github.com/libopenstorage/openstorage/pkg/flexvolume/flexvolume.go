/*
Package flexvolume implements utility code for Kubernetes flexvolumes.

https://github.com/kubernetes/kubernetes/pull/13840
https://github.com/kubernetes/kubernetes/tree/master/examples/flexvolume
*/
package flexvolume

import "encoding/json"

// Client is the client for a flexvolume implementation.
//
// It is both called from the wrapper cli tool, and implemented by a given implementation.
type Client interface {
	Init() error
	Attach(jsonOptions map[string]string) error
	Detach(mountDevice string, unmountBeforeDetach bool) error
	Mount(targetMountDir string, mountDevice string, jsonOptions map[string]string) error
	Unmount(mountDir string) error
}

// NewClient returns a new Client for the given APIClient.
func NewClient(apiClient APIClient) Client {
	return newClient(apiClient)
}

// NewLocalAPIClient returns a new APIClient for the given APIServer.
func NewLocalAPIClient(apiServer APIServer) APIClient {
	return newLocalAPIClient(apiServer)
}

// NewAPIServer returns a new APIServer for the given Client.
func NewAPIServer(client Client) APIServer {
	return newAPIServer(client)
}

// BytesToJSONOptions converts a JSON string to a map of JSON options.
func BytesToJSONOptions(value []byte) (map[string]string, error) {
	if value == nil || len(value) == 0 {
		return nil, nil
	}
	m := make(map[string]string)
	if err := json.Unmarshal(value, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// JSONOptionsToBytes converts a map of JSON Options to a JSON string.
func JSONOptionsToBytes(jsonOptions map[string]string) ([]byte, error) {
	if jsonOptions == nil || len(jsonOptions) == 0 {
		return nil, nil
	}
	return json.Marshal(jsonOptions)
}
