// Copyright 2014 Google Inc. All Rights Reserved.
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

// Handler for /api/

package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/manager"
)

const (
	ApiResource   = "/api/v1.0/"
	ContainersApi = "containers"
	MachineApi    = "machine"
)

func HandleRequest(m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	start := time.Now()

	u := r.URL

	// Get API request type.
	requestType := u.Path[len(ApiResource):]
	i := strings.Index(requestType, "/")
	requestArgs := ""
	if i != -1 {
		requestArgs = requestType[i:]
		requestType = requestType[:i]
	}

	switch {
	case requestType == MachineApi:
		log.Printf("Api - Machine")

		// Get the MachineInfo
		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			return err
		}

		out, err := json.Marshal(machineInfo)
		if err != nil {
			fmt.Fprintf(w, "Failed to marshall MachineInfo with error: %s", err)
		}
		w.Write(out)
	case requestType == ContainersApi:
		// The container name is the path after the requestType
		containerName := requestArgs

		log.Printf("Api - Container(%s)", containerName)

		var query info.ContainerInfoRequest
		decoder := json.NewDecoder(r.Body)
		err := decoder.Decode(&query)
		if err != nil && err != io.EOF {
			return fmt.Errorf("unable to decode the json value: ", err)
		}
		// Get the container.
		cont, err := m.GetContainerInfo(containerName, &query)
		if err != nil {
			fmt.Fprintf(w, "Failed to get container \"%s\" with error: %s", containerName, err)
			return err
		}

		// Only output the container as JSON.
		out, err := json.Marshal(cont)
		if err != nil {
			fmt.Fprintf(w, "Failed to marshall container %q with error: %s", containerName, err)
		}
		w.Write(out)
	default:
		return fmt.Errorf("unknown API request type %q", requestType)
	}

	log.Printf("Request took %s", time.Since(start))
	return nil
}
