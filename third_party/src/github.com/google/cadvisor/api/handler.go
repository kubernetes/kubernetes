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
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/cadvisor/manager"
)

const (
	ApiResource   = "/api/v1.0/"
	ContainersApi = "containers"
	MachineApi    = "machine"
)

func HandleRequest(m manager.Manager, w http.ResponseWriter, u *url.URL) error {
	start := time.Now()

	// Get API request type.
	requestType := u.Path[len(ApiResource):]
	i := strings.Index(requestType, "/")
	requestArgs := ""
	if i != -1 {
		requestArgs = requestType[i:]
		requestType = requestType[:i]
	}

	if requestType == MachineApi {
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
	} else if requestType == ContainersApi {
		// The container name is the path after the requestType
		containerName := requestArgs

		log.Printf("Api - Container(%s)", containerName)

		// Get the container.
		cont, err := m.GetContainerInfo(containerName)
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
	} else {
		return fmt.Errorf("unknown API request type %q", requestType)
	}

	log.Printf("Request took %s", time.Since(start))
	return nil
}
