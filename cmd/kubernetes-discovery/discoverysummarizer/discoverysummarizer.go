/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package discoverysummarizer

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	config "k8s.io/kubernetes/cmd/kubernetes-discovery/discoverysummarizer/apis/config/v1alpha1"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type DiscoverySummarizer interface {
	Run(port string) error
}

type discoverySummarizerServer struct {
	// The list of servers as read from the config file.
	serverList         config.FederatedServerList
	groupVersionPaths  map[string][]string
	legacyVersionPaths map[string][]string
}

// Ensure that discoverySummarizerServer implements DiscoverySummarizer interface.
var _ DiscoverySummarizer = &discoverySummarizerServer{}

// Creates a server to summarize all group versions
// supported by the servers mentioned in the given config file.
// Call Run() to bring up the server.
func NewDiscoverySummarizer(configFilePath string) (DiscoverySummarizer, error) {
	file, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("Error in reading config file: %v\n", err)
	}
	ds := discoverySummarizerServer{
		groupVersionPaths:  map[string][]string{},
		legacyVersionPaths: map[string][]string{},
	}
	err = json.Unmarshal(file, &ds.serverList)
	if err != nil {
		return nil, fmt.Errorf("Error in marshalling config file to json: %v\n", err)
	}

	for _, server := range ds.serverList.Servers {
		for _, groupVersionPath := range server.GroupVersionDiscoveryPaths {
			if groupVersionPath.IsLegacy {
				ds.legacyVersionPaths[groupVersionPath.Path] = append(ds.legacyVersionPaths[groupVersionPath.Path], server.ServerAddress)
			} else {
				ds.groupVersionPaths[groupVersionPath.Path] = append(ds.groupVersionPaths[groupVersionPath.Path], server.ServerAddress)
			}
		}
	}
	return &ds, nil
}

// Brings up the server at the given port.
// TODO: Add HTTPS support.
func (ds *discoverySummarizerServer) Run(port string) error {
	http.HandleFunc("/", ds.indexHandler)
	// Register a handler for all paths.
	for path := range ds.groupVersionPaths {
		p := path
		fmt.Printf("setting up a handler for %s\n", p)
		http.HandleFunc(p, ds.summarizeGroupVersionsHandler(p))
	}
	for path := range ds.legacyVersionPaths {
		p := path
		fmt.Printf("setting up a handler for %s\n", p)
		http.HandleFunc(p, ds.summarizeLegacyVersionsHandler(p))
	}
	fmt.Printf("Server running on port %s\n", port)
	return http.ListenAndServe(":"+port, nil)
}

// Handler for "/"
func (ds *discoverySummarizerServer) indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Success"))
}

// Handler for group versions summarizer.
func (ds *discoverySummarizerServer) summarizeGroupVersionsHandler(path string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var apiGroupList *unversioned.APIGroupList
		// TODO: We can cache calls to all servers.
		groups := make(chan *unversioned.APIGroupList)
		errorChannel := make(chan error)
		for _, serverAddress := range ds.groupVersionPaths[path] {
			addr := serverAddress
			go func(groups chan *unversioned.APIGroupList, error_channel chan error) {
				groupList, err := ds.getAPIGroupList(addr + path)
				if err != nil {
					errorChannel <- err
					return
				}
				groups <- groupList
				return
			}(groups, errorChannel)
		}

		var groupList *unversioned.APIGroupList
		var err error
		for range ds.groupVersionPaths[path] {
			select {
			case groupList = <-groups:
				if apiGroupList == nil {
					apiGroupList = &unversioned.APIGroupList{}
					*apiGroupList = *groupList
				} else {
					apiGroupList.Groups = append(apiGroupList.Groups, groupList.Groups...)
				}
			case err = <-errorChannel:
				ds.writeErr(http.StatusBadGateway, err, w)
				return
			}
		}

		ds.writeRawJSON(http.StatusOK, *apiGroupList, w)
		return
	}
}

// Handler for legacy versions summarizer.
func (ds *discoverySummarizerServer) summarizeLegacyVersionsHandler(path string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(ds.legacyVersionPaths[path]) > 1 {
			err := fmt.Errorf("invalid multiple servers serving legacy group %v", ds.legacyVersionPaths[path])
			ds.writeErr(http.StatusInternalServerError, err, w)
		}
		serverAddress := ds.legacyVersionPaths[path][0]
		apiVersions, err := ds.getAPIVersions(serverAddress + path)
		if err != nil {
			ds.writeErr(http.StatusBadGateway, err, w)
			return
		}
		ds.writeRawJSON(http.StatusOK, apiVersions, w)
		return
	}
}

func (ds *discoverySummarizerServer) getAPIGroupList(serverAddress string) (*unversioned.APIGroupList, error) {
	response, err := http.Get(serverAddress)
	if err != nil {
		return nil, fmt.Errorf("Error in fetching %s: %v", serverAddress, err)
	}
	defer response.Body.Close()
	contents, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Error reading response from %s: %v", serverAddress, err)
	}
	var apiGroupList unversioned.APIGroupList
	err = json.Unmarshal(contents, &apiGroupList)
	if err != nil {
		return nil, fmt.Errorf("Error in unmarshalling response from server %s: %v", serverAddress, err)
	}
	return &apiGroupList, nil
}

func (ds *discoverySummarizerServer) getAPIVersions(serverAddress string) (*unversioned.APIVersions, error) {
	response, err := http.Get(serverAddress)
	if err != nil {
		return nil, fmt.Errorf("Error in fetching %s: %v", serverAddress, err)
	}
	defer response.Body.Close()
	contents, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Error reading response from %s: %v", serverAddress, err)
	}
	var apiVersions unversioned.APIVersions
	err = json.Unmarshal(contents, &apiVersions)
	if err != nil {
		return nil, fmt.Errorf("Error in unmarshalling response from server %s: %v", serverAddress, err)
	}
	return &apiVersions, nil
}

// TODO: Pass a runtime.Object here instead of interface{} and use the encoding/decoding stack from kubernetes apiserver.
func (ds *discoverySummarizerServer) writeRawJSON(statusCode int, object interface{}, w http.ResponseWriter) {
	output, err := json.MarshalIndent(object, "", "  ")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(output)
}

func (ds *discoverySummarizerServer) writeErr(statusCode int, err error, w http.ResponseWriter) {
	http.Error(w, err.Error(), statusCode)
}
