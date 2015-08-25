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

package main

import (
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

const (
	badRequest                = "Bad request. Not a POST request"
	unknownFunction           = "unknown function"
	incorrectFunctionArgument = "incorrect function argument"
	notGivenFunctionArgument  = "not given function argument"
	consumeCPUAddress         = "/ConsumeCPU"
	consumeMemAddress         = "/ConsumeMem"
	getCurrentStatusAddress   = "/GetCurrentStatus"
	milicoresQuery            = "milicores"
	megabytesQuery            = "megabytes"
	durationSecQuery          = "durationSec"
)

type ResourceConsumerHandler struct{}

func (handler ResourceConsumerHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(w, badRequest, http.StatusBadRequest)
	}
	// parsing POST request data and URL data
	if err := req.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// handle consumeCPU
	if req.URL.Path == consumeCPUAddress {
		handler.handleConsumeCPU(w, req.Form)
		return
	}
	// handle consumeMem
	if req.URL.Path == consumeMemAddress {
		handler.handleConsumeMem(w, req.Form)
		return
	}
	// handle getCurrentStatus
	if req.URL.Path == getCurrentStatusAddress {
		handler.handleGetCurrentStatus(w)
		return
	}
	http.Error(w, unknownFunction, http.StatusNotFound)
}

func (handler ResourceConsumerHandler) handleConsumeCPU(w http.ResponseWriter, query url.Values) {
	// geting string data for consumeCPU
	durationSecString := query.Get(durationSecQuery)
	milicoresString := query.Get(milicoresQuery)
	if durationSecString == "" || milicoresString == "" {
		http.Error(w, notGivenFunctionArgument, http.StatusBadRequest)
		return
	} else {
		// convert data (strings to ints) for consumeCPU
		durationSec, durationSecError := strconv.Atoi(durationSecString)
		milicores, milicoresError := strconv.Atoi(milicoresString)
		if durationSecError != nil || milicoresError != nil {
			http.Error(w, incorrectFunctionArgument, http.StatusBadRequest)
			return
		}
		go ConsumeCPU(milicores, durationSec)
		fmt.Fprintln(w, consumeCPUAddress[1:])
		fmt.Fprintln(w, milicores, milicoresQuery)
		fmt.Fprintln(w, durationSec, durationSecQuery)

	}

}

func (handler ResourceConsumerHandler) handleConsumeMem(w http.ResponseWriter, query url.Values) {
	// geting string data for consumeMem
	durationSecString := query.Get(durationSecQuery)
	megabytesString := query.Get(megabytesQuery)
	if durationSecString == "" || megabytesString == "" {
		http.Error(w, notGivenFunctionArgument, http.StatusBadRequest)
		return
	} else {
		// convert data (strings to ints) for consumeMem
		durationSec, durationSecError := strconv.Atoi(durationSecString)
		megabytes, megabytesError := strconv.Atoi(megabytesString)
		if durationSecError != nil || megabytesError != nil {
			http.Error(w, incorrectFunctionArgument, http.StatusBadRequest)
			return
		}
		ConsumeMem(megabytes, durationSec)
		fmt.Fprintln(w, "Warning: not implemented!")
		fmt.Fprintln(w, consumeMemAddress[1:])
		fmt.Fprintln(w, megabytes, megabytesQuery)
		fmt.Fprintln(w, durationSec, durationSecQuery)
	}
}

func (handler ResourceConsumerHandler) handleGetCurrentStatus(w http.ResponseWriter) {
	GetCurrentStatus()
	fmt.Fprintln(w, "Warning: not implemented!")
	fmt.Fprint(w, getCurrentStatusAddress[1:])
}
