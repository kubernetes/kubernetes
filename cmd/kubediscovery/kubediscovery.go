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
package main

import (
	"log"
	"net/http"
	"os"

	kd "k8s.io/kubernetes/pkg/kubediscovery"
)

func main() {

	// Make sure the CA cert for the cluster exists and is readable.
	// We are expecting a base64 encoded version of the cert PEM as this is how
	// the cert would most likely be provided via kubernetes secrets.
	if _, err := os.Stat(kd.CAPath); os.IsNotExist(err) {
		log.Fatalf("CA does not exist: %s", kd.CAPath)
	}
	// Test read permissions
	file, err := os.Open(kd.CAPath)
	if err != nil {
		log.Fatalf("ERROR: Unable to read %s", kd.CAPath)
	}
	file.Close()

	router := kd.NewRouter()
	log.Printf("Listening for requests on port 9898.")
	log.Fatal(http.ListenAndServe(":9898", router))
}
