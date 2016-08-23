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
package kubediscovery

import (
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/square/go-jose"
)

// TODO: Just using a hardcoded token for now.
const tempTokenId string = "TOKENID"
const tempToken string = "EF1BA4F26DDA9FE2"

// CAPath is the expected location of our cluster's CA to be distributed to
// clients looking to connect. Because we expect to use kubernetes secrets
// for the time being, this file is expected to be a base64 encoded version
// of the normal cert PEM.
const CAPath = "/tmp/secret/ca.pem"

// tokenLoader is an interface for abstracting how we validate
// token IDs and lookup their corresponding token.
type tokenLoader interface {
	// Lookup returns the token for a given token ID, or an error if the token ID
	// does not exist. Both token and it's ID are expected to be hex encoded strings.
	Lookup(tokenId string) (string, error)
}

type hardcodedTokenLoader struct {
}

func (tl *hardcodedTokenLoader) Lookup(tokenId string) (string, error) {
	if tokenId == tempTokenId {
		return tempToken, nil
	}
	return "", errors.New(fmt.Sprintf("invalid token: %s", tokenId))
}

// caLoader is an interface for abstracting how we load the CA certificates
// for the cluster.
type caLoader interface {
	LoadPEM() (string, error)
}

// fsCALoader is a caLoader for loading the PEM encoded CA from
// /tmp/secret/ca.pem.
type fsCALoader struct {
}

func (cl *fsCALoader) LoadPEM() (string, error) {
	file, err := os.Open(CAPath)
	if err != nil {
		return "", err
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// ClusterInfoHandler implements the http.ServeHTTP method and allows us to
// mock out portions of the request handler in tests.
type ClusterInfoHandler struct {
	tokenLoader tokenLoader
	caLoader    caLoader
}

func NewClusterInfoHandler() *ClusterInfoHandler {
	tl := hardcodedTokenLoader{}
	cl := fsCALoader{}
	return &ClusterInfoHandler{
		tokenLoader: &tl,
		caLoader:    &cl,
	}
}

func (cih *ClusterInfoHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	tokenId := req.FormValue("token-id")
	log.Printf("Got token ID: %s", tokenId)
	token, err := cih.tokenLoader.Lookup(tokenId)
	if err != nil {
		log.Printf("Invalid token: %s", err)
		http.Error(resp, "Forbidden", http.StatusForbidden)
		return
	}
	log.Printf("Loaded token: %s", token)

	caPEM, err := cih.caLoader.LoadPEM()
	caB64 := base64.StdEncoding.EncodeToString([]byte(caPEM))

	if err != nil {
		http.Error(resp, "Error encoding CA", http.StatusInternalServerError)
		return
	}

	clusterInfo := ClusterInfo{
		Type:             "ClusterInfo",
		Version:          "v1",
		RootCertificates: caB64,
	}

	// Instantiate an signer using HMAC-SHA256.
	hmacTestKey := fromHexBytes(token)
	signer, err := jose.NewSigner(jose.HS256, hmacTestKey)
	if err != nil {
		http.Error(resp, fmt.Sprintf("Error creating JWS signer: %s", err), http.StatusInternalServerError)
		return
	}

	payload, err := json.Marshal(clusterInfo)
	if err != nil {
		http.Error(resp, fmt.Sprintf("Error serializing clusterInfo to JSON: %s", err),
			http.StatusInternalServerError)
		return
	}

	// Sign a sample payload. Calling the signer returns a protected JWS object,
	// which can then be serialized for output afterwards. An error would
	// indicate a problem in an underlying cryptographic primitive.
	jws, err := signer.Sign(payload)
	if err != nil {
		http.Error(resp, fmt.Sprintf("Error signing clusterInfo to JSON: %s", err),
			http.StatusInternalServerError)
		return
	}

	// Serialize the encrypted object using the full serialization format.
	// Alternatively you can also use the compact format here by calling
	// object.CompactSerialize() instead.
	serialized := jws.FullSerialize()

	resp.Write([]byte(serialized))

}

// TODO: Move into test package
// TODO: Should we use base64 instead?
func fromHexBytes(base16 string) []byte {
	val, err := hex.DecodeString(base16)
	if err != nil {
		panic(fmt.Sprintf("Invalid test data: %s", err))
	}
	return val
}
