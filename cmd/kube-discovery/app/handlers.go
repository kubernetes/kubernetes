/*
Copyright 2014 The Kubernetes Authors.

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

package discovery

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/square/go-jose"
)

const secretPath = "/tmp/secret"

// CAPath is the expected location of our cluster's CA to be distributed to
// clients looking to connect. Because we expect to use kubernetes secrets
// for the time being, this file is expected to be a base64 encoded version
// of the normal cert PEM.
const CAPath = secretPath + "/ca.pem"

// caLoader is an interface for abstracting how we load the CA certificates
// for the cluster.
type caLoader interface {
	LoadPEM() (string, error)
}

// fsCALoader is a caLoader for loading the PEM encoded CA from
// /tmp/secret/ca.pem.
type fsCALoader struct {
	certData string
}

func (cl *fsCALoader) LoadPEM() (string, error) {
	if cl.certData == "" {
		data, err := ioutil.ReadFile(CAPath)
		if err != nil {
			return "", err
		}

		cl.certData = string(data)
	}

	return cl.certData, nil
}

const TokenMapPath = secretPath + "/token-map.json"
const EndpointListPath = secretPath + "/endpoint-list.json"

// tokenLoader is an interface for abstracting how we validate
// token IDs and lookup their corresponding token.
type tokenLoader interface {
	// Lookup returns the token for a given token ID, or an error if the token ID
	// does not exist. Both token and it's ID are expected be strings.
	LoadAndLookup(tokenID string) (string, error)
}

type jsonFileTokenLoader struct {
	tokenMap map[string]string
}

func (tl *jsonFileTokenLoader) LoadAndLookup(tokenID string) (string, error) {
	if len(tl.tokenMap) == 0 {
		data, err := ioutil.ReadFile(TokenMapPath)
		if err != nil {
			return "", err
		}
		if err := json.Unmarshal(data, &tl.tokenMap); err != nil {
			return "", err
		}
	}
	if val, ok := tl.tokenMap[tokenID]; ok {
		return val, nil
	}
	return "", errors.New(fmt.Sprintf("invalid token: %s", tokenID))
}

type endpointsLoader interface {
	LoadList() ([]string, error)
}

type jsonFileEndpointsLoader struct {
	endpoints []string
}

func (el *jsonFileEndpointsLoader) LoadList() ([]string, error) {
	if len(el.endpoints) == 0 {
		data, err := ioutil.ReadFile(EndpointListPath)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &el.endpoints); err != nil {
			return nil, err
		}
	}
	return el.endpoints, nil
}

// ClusterInfoHandler implements the http.ServeHTTP method and allows us to
// mock out portions of the request handler in tests.
type ClusterInfoHandler struct {
	tokenLoader     tokenLoader
	caLoader        caLoader
	endpointsLoader endpointsLoader
}

func NewClusterInfoHandler() *ClusterInfoHandler {
	return &ClusterInfoHandler{
		tokenLoader:     &jsonFileTokenLoader{},
		caLoader:        &fsCALoader{},
		endpointsLoader: &jsonFileEndpointsLoader{},
	}
}

func (cih *ClusterInfoHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	tokenID := req.FormValue("token-id")
	log.Printf("Got token ID: %s", tokenID)
	token, err := cih.tokenLoader.LoadAndLookup(tokenID)
	if err != nil {
		log.Print(err)
		http.Error(resp, "Forbidden", http.StatusForbidden)
		return
	}
	log.Printf("Loaded token: %s", token)

	// TODO probably should not leak server-side errors to the client
	caPEM, err := cih.caLoader.LoadPEM()
	if err != nil {
		err = fmt.Errorf("Error loading root CA certificate data: %s", err)
		log.Println(err)
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}
	log.Printf("Loaded CA: %s", caPEM)

	endpoints, err := cih.endpointsLoader.LoadList()
	if err != nil {
		err = fmt.Errorf("Error loading list of API endpoints: %s", err)
		log.Println(err)
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}

	clusterInfo := ClusterInfo{
		CertificateAuthorities: []string{caPEM},
		Endpoints:              endpoints,
	}

	// Instantiate an signer using HMAC-SHA256.
	hmacKey := []byte(token)

	log.Printf("Key is %d bytes long", len(hmacKey))
	signer, err := jose.NewSigner(jose.HS256, hmacKey)
	if err != nil {
		err = fmt.Errorf("Error creating JWS signer: %s", err)
		log.Println(err)
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}

	payload, err := json.Marshal(clusterInfo)
	if err != nil {
		err = fmt.Errorf("Error serializing clusterInfo to JSON: %s", err)
		log.Println(err)
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}

	// Sign a sample payload. Calling the signer returns a protected JWS object,
	// which can then be serialized for output afterwards. An error would
	// indicate a problem in an underlying cryptographic primitive.
	jws, err := signer.Sign(payload)
	if err != nil {
		err = fmt.Errorf("Error signing clusterInfo with JWS: %s", err)
		log.Println(err)
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}

	// Serialize the encrypted object using the full serialization format.
	// Alternatively you can also use the compact format here by calling
	// object.CompactSerialize() instead.
	serialized := jws.FullSerialize()

	resp.Write([]byte(serialized))

}
