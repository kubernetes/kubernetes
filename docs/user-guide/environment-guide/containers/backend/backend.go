/*
Copyright 2015 The Kubernetes Authors.

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
	"log"
	"net/http"
	"os"
)

func printInfo(resp http.ResponseWriter, req *http.Request) {
	name := os.Getenv("POD_NAME")
	namespace := os.Getenv("POD_NAMESPACE")
	fmt.Fprintf(resp, "Backend Container\n")
	fmt.Fprintf(resp, "Backend Pod Name: %v\n", name)
	fmt.Fprintf(resp, "Backend Namespace: %v\n", namespace)
}

func main() {
	http.HandleFunc("/", printInfo)
	log.Fatal(http.ListenAndServe(":5000", nil))
}
