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

package main

import (
	"log"
	"net/http"
	"os"

	kd "k8s.io/kubernetes/cmd/kube-discovery/app"
)

func main() {
	// Make sure we can load critical files, and be nice to the user by
	// printing descriptive error message when we fail.
	for desc, path := range map[string]string{
		"root CA certificate":   kd.CAPath,
		"token map file":        kd.TokenMapPath,
		"list of API endpoints": kd.EndpointListPath,
	} {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			log.Fatalf("%s does not exist: %s", desc, path)
		}
		// Test read permissions
		file, err := os.Open(path)
		if err != nil {
			log.Fatalf("Unable to open %s (%q [%s])", desc, path, err)
		}
		file.Close()
	}

	router := kd.NewRouter()
	log.Printf("Listening for requests on port 9898.")
	log.Fatal(http.ListenAndServe(":9898", router))
}
