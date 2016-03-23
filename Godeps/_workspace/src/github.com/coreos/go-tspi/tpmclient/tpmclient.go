// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tpmclient

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/coreos/go-tspi/verification"
	"github.com/coreos/go-tspi/tspiconst"
)

// TPMClient represents a connection to a system running a daemon providing
// access to TPM functionality
type TPMClient struct {
	host    string
	timeout time.Duration
}

const GetEKCertURL = "/v1/getEkcert"
const ExtendURL = "/v1/extend"
const QuoteURL = "/v1/quote"
const GenerateAikURL = "/v1/generateAik"
const GenerateKeyURL = "/v1/generateKey"
const AikChallengeURL = "/v1/aikChallenge"

func (client *TPMClient) get(endpoint string) (*http.Response, error) {
	url := fmt.Sprintf("http://%s%s", client.host, endpoint)
	httpClient := &http.Client{
		Timeout: client.timeout,
	}
	resp, err := httpClient.Get(url)
	return resp, err
}

func (client *TPMClient) post(endpoint string, data io.Reader) (*http.Response, error) {
	url := fmt.Sprintf("http://%s%s", client.host, endpoint)
	httpClient := &http.Client{
		Timeout: client.timeout,
	}
	resp, err := httpClient.Post(url, "application/json", data)
	return resp, err
}

type EkcertResponse struct {
	EKCert []byte
}

// GetEKCert obtains the Endorsement Key certificate from the client TPM. This
// is an X509 certificate containing the public half of the Endorsement Key
// and a signature chain chaining back to a vendor-issued signing certificate.
func (client *TPMClient) GetEKCert() (ekcert []byte, err error) {
	var ekcertData EkcertResponse

	ekresp, err := client.get(GetEKCertURL)
	if err != nil {
		return nil, fmt.Errorf("Can't obtain ekcert: %s", err)
	}
	defer ekresp.Body.Close()
	body, err := ioutil.ReadAll(ekresp.Body)
	if err != nil {
		return nil, fmt.Errorf("Can't read ekcert response: %s", err)
	}

	err = json.Unmarshal(body, &ekcertData)
	if err != nil {
		return nil, fmt.Errorf("Can't parse ekcert response: %s", err)
	}

	return ekcertData.EKCert, nil
}

type AikResponse struct {
	AIKBlob []byte
	AIKPub  []byte
}

// GenerateAIK requests that the TPM generate a new Attestation Identity Key.
// It returns an unencrypted copy of the public half of the AIK, along with
// a TSPI key blob encrypted by the TPM.
func (client *TPMClient) GenerateAIK() (aikpub []byte, aikblob []byte, err error) {
	var aikData AikResponse

	aikresp, err := client.post(GenerateAikURL, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't generate AIK: %s", err)
	}
	defer aikresp.Body.Close()
	body, err := ioutil.ReadAll(aikresp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't read AIK response: %s", err)
	}

	err = json.Unmarshal(body, &aikData)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't parse AIK response: %s (%s)", err, body)
	}

	aikpub = aikData.AIKPub
	aikblob = aikData.AIKBlob

	return aikpub, aikblob, nil
}

type KeyData struct {
	KeyFlags int
}

type KeyResponse struct {
	KeyBlob	[]byte
	KeyPub	[]byte
}

// GenerateKey requests that the TPM generate a new keypair
func (client *TPMClient) GenerateKey(flags int) (keypub []byte, keyblob []byte, err error) {
	var keyData KeyData
	var keyResponse KeyResponse

	keyData.KeyFlags = flags
	request, err := json.Marshal(keyData)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't construct request JSON: %s", err)
	}

	keyresp, err := client.post(GenerateKeyURL, bytes.NewBuffer(request))
	if err != nil {
		return nil, nil, fmt.Errorf("Can't generate key: %s", err)
	}
	defer keyresp.Body.Close()
	body, err := ioutil.ReadAll(keyresp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't read key response: %s", err)
	}

	err = json.Unmarshal(body, &keyResponse)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't parse key response: %s (%s)", err, body)
	}

	keypub = keyResponse.KeyPub
	keyblob = keyResponse.KeyBlob

	return keypub, keyblob, nil
}

type ChallengeData struct {
	AIK     []byte
	Asymenc []byte
	Symenc  []byte
}

type ChallengeResponse struct {
	Response []byte
}

// ValidateAIK challenges the TPM to validate an AIK by using the provided
// key blob to decrypt a secret encrypted with the public half of the
// AIK. This will only be possible if the TPM is able to decrypt the
// encrypted key blob.  The AIK is used to decrypt asymenc, which then
// provides the AES key used to encrypt symenc. Decrypting symenc provides
// the original secret, which is then returned.
func (client *TPMClient) ValidateAIK(aikblob []byte, asymenc []byte, symenc []byte) (secret []byte, err error) {
	var challenge ChallengeData
	var response ChallengeResponse

	challenge.AIK = aikblob
	challenge.Asymenc = asymenc
	challenge.Symenc = symenc

	request, err := json.Marshal(challenge)
	if err != nil {
		return nil, fmt.Errorf("Can't construct challenge JSON: %s", err)
	}
	chalresp, err := client.post(AikChallengeURL, bytes.NewBuffer(request))
	if err != nil {
		return nil, fmt.Errorf("Can't perform AIK challenge: %s", err)
	}
	defer chalresp.Body.Close()
	body, err := ioutil.ReadAll(chalresp.Body)
	if err != nil {
		return nil, fmt.Errorf("Can't read AIK challenge response: %s", err)
	}

	err = json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("Can't parse AIK challenge response: %s", err)
	}
	return response.Response, nil
}

type ExtendInput struct {
	Pcr       int
	Eventtype int
	Data      []byte
	Event     string
}

// Extend extends a TPM PCR with the provided data. If event is nil, data must
// be pre-hashed with SHA1 and will be used to extend the PCR directly. If
// event is not nil, data and event will be hashed to generate the extension
// value. Event will then be stored in the TPM event log.
func (client *TPMClient) Extend(pcr int, eventtype int, data []byte, event string) error {
	var extendData ExtendInput

	extendData.Pcr = pcr
	extendData.Eventtype = eventtype
	extendData.Data = data
	extendData.Event = event

	request, err := json.Marshal(extendData)
	if err != nil {
		return fmt.Errorf("Can't construct extension JSON: %s", err)
	}
	chalresp, err := client.post(ExtendURL, bytes.NewBuffer(request))
	if err != nil {
		return fmt.Errorf("Can't perform PCR extension: %s", err)
	}
	defer chalresp.Body.Close()

	return nil
}

type QuoteData struct {
	AIK   []byte
	PCRs  []int
	Nonce []byte
}

type QuoteResponse struct {
	Data       []byte
	Validation []byte
	PCRValues  [][]byte
	Events     []tspiconst.Log
}

// GetQuote obtains a PCR quote from the TPM. It takes the aikpub Tspi Key, the
// encrypted AIK blob and a list of PCRs as arguments. The response will
// contain an array of PCR values, an array of log entries and any error.
func (client *TPMClient) GetQuote(aikpub []byte, aikblob []byte, pcrs []int) (pcrvals [][]byte, log []tspiconst.Log, err error) {
	var quoteRequest QuoteData
	var response QuoteResponse

	nonce := make([]byte, 20)
	_, err = rand.Read(nonce)
	if err != nil {
		return nil, nil, fmt.Errorf("Unable to generate nonce: %s", err)
	}

	quoteRequest.AIK = aikblob
	quoteRequest.PCRs = pcrs
	quoteRequest.Nonce = nonce

	request, err := json.Marshal(quoteRequest)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't construct quote request JSON: %s", err)
	}
	chalresp, err := client.post(QuoteURL, bytes.NewBuffer(request))
	if err != nil {
		return nil, nil, fmt.Errorf("Can't perform obtain quote: %s", err)
	}
	defer chalresp.Body.Close()
	body, err := ioutil.ReadAll(chalresp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't read quote response: %s", err)
	}

	err = json.Unmarshal(body, &response)
	if err != nil {
		return nil, nil, fmt.Errorf("Can't parse quote response: %s", err)
	}

	aikmod := aikpub[28:]

	err = verification.QuoteVerify(response.Data, response.Validation, aikmod, response.PCRValues, nonce)

	if err != nil {
		return nil, nil, fmt.Errorf("Can't verify quote: %s", err)
	}

	return response.PCRValues, response.Events, nil
}

// New returns a TPMClient structure configured to connect to the provided
// host with the provided timeout.
func New(host string, timeout time.Duration) *TPMClient {
	return &TPMClient{
		host:    host,
		timeout: timeout,
	}
}
