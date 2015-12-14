/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
)

type restService struct {
	config ConcertoConfig
	client *http.Client
}

func newRestService(config ConcertoConfig) (concertoRESTService, error) {
	client, err := httpClient(config)
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP client: %v", err)
	}
	return &restService{config, client}, nil
}

func httpClient(config ConcertoConfig) (*http.Client, error) {
	// load client certificate
	cert, err := tls.LoadX509KeyPair(config.Connection.Cert, config.Connection.Key)
	if err != nil {
		return nil, fmt.Errorf("error loading X509 key pair: %v", err)
	}
	// load CA file to verify server
	CA_Pool := x509.NewCertPool()
	severCert, err := ioutil.ReadFile(config.Connection.CACert)
	if err != nil {
		return nil, fmt.Errorf("could not load CA file: %v", err)
	}
	CA_Pool.AppendCertsFromPEM(severCert)
	// create a client with specific transport configurations
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:      CA_Pool,
			Certificates: []tls.Certificate{cert},
		},
	}
	client := &http.Client{Transport: transport}

	return client, nil
}

func (r *restService) Post(path string, json []byte) ([]byte, int, error) {
	output := strings.NewReader(string(json))
	response, err := r.client.Post(r.config.Connection.APIEndpoint+path, "application/json", output)
	if err != nil {
		return nil, -1, fmt.Errorf("error on http request (POST %v): %v", r.config.Connection.APIEndpoint+path, err)
	}
	defer response.Body.Close()

	body, _ := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, -1, fmt.Errorf("error reading http request body: %v", err)
	}

	return body, response.StatusCode, err
}

func (r *restService) Delete(path string) ([]byte, int, error) {
	request, err := http.NewRequest("DELETE", r.config.Connection.APIEndpoint+path, nil)
	if err != nil {
		return nil, -1, fmt.Errorf("error creating http request (DELETE %v): %v", r.config.Connection.APIEndpoint+path, err)
	}
	response, err := r.client.Do(request)
	if err != nil {
		return nil, -1, fmt.Errorf("error executing http request (DELETE %v): %v", r.config.Connection.APIEndpoint+path, err)
	}
	defer response.Body.Close()

	body, _ := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, -1, fmt.Errorf("error reading http request body: %v", err)
	}

	return body, response.StatusCode, nil
}

func (r *restService) Get(path string) ([]byte, int, error) {
	response, err := r.client.Get(r.config.Connection.APIEndpoint + path)
	if err != nil {
		return nil, -1, fmt.Errorf("error on http request (GET %v): %v", r.config.Connection.APIEndpoint+path, err)
	}
	defer response.Body.Close()

	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, -1, fmt.Errorf("error reading http request body: %v", err)
	}

	return body, response.StatusCode, nil
}
