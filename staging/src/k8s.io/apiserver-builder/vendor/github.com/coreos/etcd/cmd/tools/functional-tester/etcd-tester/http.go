// Copyright 2015 The etcd Authors
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
	"net/http"
)

type statusHandler struct {
	status *Status
}

func (sh statusHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	en := json.NewEncoder(w)

	sh.status.mu.Lock()
	defer sh.status.mu.Unlock()

	if err := en.Encode(Status{
		Since:      sh.status.Since,
		Failures:   sh.status.Failures,
		RoundLimit: sh.status.RoundLimit,
		Cluster:    sh.status.cluster.Status(),
		cluster:    sh.status.cluster,
		Round:      sh.status.Round,
		Case:       sh.status.Case,
	}); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
