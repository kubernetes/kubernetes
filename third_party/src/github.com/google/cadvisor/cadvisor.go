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

package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/google/cadvisor/api"
	"github.com/google/cadvisor/container/docker"
	"github.com/google/cadvisor/container/lmctfy"
	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/pages"
	"github.com/google/cadvisor/pages/static"
)

var argPort = flag.Int("port", 8080, "port to listen")
var argAllowLmctfy = flag.Bool("allow_lmctfy", true, "whether to allow lmctfy as a container handler")

var argDbDriver = flag.String("storage_driver", "memory", "storage driver to use. Options are: memory (default) and influxdb")

func main() {
	flag.Parse()

	storageDriver, err := NewStorageDriver(*argDbDriver)
	if err != nil {
		log.Fatalf("Failed to connect to database: %s", err)
	}

	containerManager, err := manager.New(storageDriver)
	if err != nil {
		log.Fatalf("Failed to create a Container Manager: %s", err)
	}

	// Register lmctfy for the root if allowed and available.
	registeredRoot := false
	if *argAllowLmctfy {
		if err := lmctfy.Register("/"); err != nil {
			log.Printf("lmctfy registration failed: %v.", err)
			log.Print("Running in docker only mode.")
		} else {
			registeredRoot = true
		}
	}

	// Register Docker for root if we were unable to register lmctfy.
	if !registeredRoot {
		if err := docker.Register(containerManager, "/"); err != nil {
			log.Printf("Docker registration failed: %v.", err)
			log.Fatalf("Unable to continue without root handler.")
		}
	}

	// Register Docker for all Docker containers.
	if err := docker.Register(containerManager, "/docker"); err != nil {
		// Ignore this error because we should work with lmctfy only
		log.Printf("Docker registration failed: %v.", err)
		log.Print("Running in lmctfy only mode.")
	}

	// Handler for static content.
	http.HandleFunc(static.StaticResource, func(w http.ResponseWriter, r *http.Request) {
		err := static.HandleRequest(w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	})

	// Handler for the API.
	http.HandleFunc(api.ApiResource, func(w http.ResponseWriter, r *http.Request) {
		err := api.HandleRequest(containerManager, w, r)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	})

	// Redirect / to containers page.
	http.Handle("/", http.RedirectHandler(pages.ContainersPage, http.StatusTemporaryRedirect))

	// Register the handler for the containers page.
	http.HandleFunc(pages.ContainersPage, func(w http.ResponseWriter, r *http.Request) {
		err := pages.ServerContainersPage(containerManager, w, r.URL)
		if err != nil {
			fmt.Fprintf(w, "%s", err)
		}
	})

	go containerManager.Start()

	log.Printf("Starting cAdvisor version: %q", info.VERSION)
	log.Print("About to serve on port ", *argPort)

	addr := fmt.Sprintf(":%v", *argPort)
	log.Fatal(http.ListenAndServe(addr, nil))
}
