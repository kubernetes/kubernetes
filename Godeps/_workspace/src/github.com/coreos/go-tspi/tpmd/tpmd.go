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

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"sync"

	"github.com/coreos/go-tspi/attestation"
	"github.com/coreos/go-tspi/tpmclient"
	"github.com/coreos/go-tspi/tspi"
	"github.com/coreos/go-tspi/tspiconst"
)

var wellKnown [20]byte

var pcrmutex sync.RWMutex

func setupContext() (*tspi.Context, *tspi.TPM, error) {
	context, err := tspi.NewContext()
	if err != nil {
		return nil, nil, err
	}

	context.Connect()
	tpm := context.GetTPM()
	tpmpolicy, err := context.CreatePolicy(tspiconst.TSS_POLICY_USAGE)
	if err != nil {
		return nil, nil, err
	}
	tpm.AssignPolicy(tpmpolicy)
	tpmpolicy.SetSecret(tspiconst.TSS_SECRET_MODE_SHA1, wellKnown[:])

	return context, tpm, nil
}

func cleanupContext(context *tspi.Context) {
	context.Close()
}

func loadSRK(context *tspi.Context) (*tspi.Key, error) {
	srk, err := context.LoadKeyByUUID(tspiconst.TSS_PS_TYPE_SYSTEM, tspi.TSS_UUID_SRK)
	if err != nil {
		return nil, err
	}

	srkpolicy, err := srk.GetPolicy(tspiconst.TSS_POLICY_USAGE)
	if err != nil {
		return nil, err
	}
	srkpolicy.SetSecret(tspiconst.TSS_SECRET_MODE_SHA1, wellKnown[:])

	return srk, nil
}

func getEkcert(rw http.ResponseWriter, request *http.Request) {
	var output tpmclient.EkcertResponse

	context, _, err := setupContext()
	defer cleanupContext(context)

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	if request.Method != "GET" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	ekcert, err := attestation.GetEKCert(context)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}

	output.EKCert = ekcert
	jsonresponse, err := json.Marshal(output)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}
	rw.Write(jsonresponse)
}

func generateAik(rw http.ResponseWriter, request *http.Request) {
	var output tpmclient.AikResponse

	context, _, err := setupContext()
	defer cleanupContext(context)

	if request.Method != "POST" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	aikpub, aikblob, err := attestation.CreateAIK(context)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	output.AIKPub = aikpub
	output.AIKBlob = aikblob

	jsonresponse, err := json.Marshal(output)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	rw.Write(jsonresponse)
}

func generateKey(rw http.ResponseWriter, request *http.Request) {
	var input tpmclient.KeyData
	var output tpmclient.KeyResponse

	body, err := ioutil.ReadAll(request.Body)

	if request.Method != "POST" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	err = json.Unmarshal(body, &input)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}

	context, _, err := setupContext()
	defer cleanupContext(context)

	srk, err := loadSRK(context)

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	key, err := context.CreateKey(input.KeyFlags)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	err = key.GenerateKey(srk)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	keypub, err := key.GetPubKeyBlob()
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	keyblob, err := key.GetKeyBlob()
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	output.KeyPub = keypub
	output.KeyBlob = keyblob

	jsonresponse, err := json.Marshal(output)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	rw.Write(jsonresponse)
}

func aikChallenge(rw http.ResponseWriter, request *http.Request) {
	body, err := ioutil.ReadAll(request.Body)
	var input tpmclient.ChallengeData
	var output tpmclient.ChallengeResponse

	context, _, err := setupContext()
	defer cleanupContext(context)

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	if request.Method != "POST" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	err = json.Unmarshal(body, &input)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}

	response, err := attestation.AIKChallengeResponse(context, input.AIK, input.Asymenc, input.Symenc)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	output.Response = response
	jsonresponse, err := json.Marshal(output)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	rw.Write(jsonresponse)
}

func quote(rw http.ResponseWriter, request *http.Request) {
	body, err := ioutil.ReadAll(request.Body)
	var input tpmclient.QuoteData
	var output tpmclient.QuoteResponse

	context, tpm, err := setupContext()
	defer cleanupContext(context)

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	if request.Method != "POST" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	err = json.Unmarshal(body, &input)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}

	pcrs, err := context.CreatePCRs(tspiconst.TSS_PCRS_STRUCT_INFO)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	err = pcrs.SetPCRs(input.PCRs)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	srk, err := context.LoadKeyByUUID(tspiconst.TSS_PS_TYPE_SYSTEM, tspi.TSS_UUID_SRK)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	srkpolicy, err := srk.GetPolicy(tspiconst.TSS_POLICY_USAGE)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	srkpolicy.SetSecret(tspiconst.TSS_SECRET_MODE_SHA1, wellKnown[:])

	aik, err := context.LoadKeyByBlob(srk, input.AIK)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	pcrmutex.Lock()
	data, validation, err := tpm.GetQuote(aik, pcrs, input.Nonce)
	if err != nil {
		pcrmutex.Unlock()
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	pcrvalues, err := pcrs.GetPCRValues()
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		pcrmutex.Unlock()
		return
	}

	log, err := tpm.GetEventLog()
	pcrmutex.Unlock()
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	output.Data = data
	output.Validation = validation
	output.PCRValues = pcrvalues
	output.Events = log

	jsonoutput, err := json.Marshal(output)
	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	rw.Write(jsonoutput)
}

func extend(rw http.ResponseWriter, request *http.Request) {
	body, err := ioutil.ReadAll(request.Body)
	var data tpmclient.ExtendInput

	context, tpm, err := setupContext()
	defer cleanupContext(context)

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}

	if request.Method != "POST" {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}

	err = json.Unmarshal(body, &data)

	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		rw.Write([]byte(err.Error()))
		return
	}

	pcrmutex.Lock()
	err = tpm.ExtendPCR(data.Pcr, data.Data, data.Eventtype, []byte(data.Event))
	pcrmutex.Unlock()

	if err != nil {
		rw.WriteHeader(http.StatusInternalServerError)
		rw.Write([]byte(err.Error()))
		return
	}
	rw.Write([]byte("OK"))
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s port\n", os.Args[0])
		return
	}
	socket := fmt.Sprintf(":%s", os.Args[1])
	http.HandleFunc(tpmclient.ExtendURL, extend)
	http.HandleFunc(tpmclient.QuoteURL, quote)
	http.HandleFunc(tpmclient.GetEKCertURL, getEkcert)
	http.HandleFunc(tpmclient.GenerateAikURL, generateAik)
	http.HandleFunc(tpmclient.GenerateKeyURL, generateKey)
	http.HandleFunc(tpmclient.AikChallengeURL, aikChallenge)
	err := http.ListenAndServe(socket, nil)
	if err != nil {
		fmt.Printf("Unable to listen - %s\n", err)
	}
}
