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
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
)

func getKubeEnv() (map[string]string, error) {
	environS := os.Environ()
	environ := make(map[string]string)
	for _, val := range environS {
		split := strings.Split(val, "=")
		if len(split) != 2 {
			return environ, fmt.Errorf("Some weird env vars")
		}
		environ[split[0]] = split[1]
	}
	for key := range environ {
		if !(strings.HasSuffix(key, "_SERVICE_HOST") ||
			strings.HasSuffix(key, "_SERVICE_PORT")) {
			delete(environ, key)
		}
	}
	return environ, nil
}

func printInfo(resp http.ResponseWriter, req *http.Request) {
	kubeVars, err := getKubeEnv()
	if err != nil {
		http.Error(resp, err.Error(), http.StatusInternalServerError)
		return
	}

	backendHost := os.Getenv("BACKEND_SRV_SERVICE_HOST")
	backendPort := os.Getenv("BACKEND_SRV_SERVICE_PORT")
	backendRsp, backendErr := http.Get(fmt.Sprintf(
		"http://%v:%v/",
		backendHost,
		backendPort))
	if backendErr == nil {
		defer backendRsp.Body.Close()
	}

	name := os.Getenv("POD_NAME")
	namespace := os.Getenv("POD_NAMESPACE")
	fmt.Fprintf(resp, "Pod Name: %v \n", name)
	fmt.Fprintf(resp, "Pod Namespace: %v \n", namespace)

	envvar := os.Getenv("USER_VAR")
	fmt.Fprintf(resp, "USER_VAR: %v \n", envvar)

	fmt.Fprintf(resp, "\nKubernetes environment variables\n")
	var keys []string
	for key := range kubeVars {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(resp, "%v = %v \n", key, kubeVars[key])
	}

	fmt.Fprintf(resp, "\nFound backend ip: %v port: %v\n", backendHost, backendPort)
	if backendErr == nil {
		fmt.Fprintf(resp, "Response from backend\n")
		io.Copy(resp, backendRsp.Body)
	} else {
		fmt.Fprintf(resp, "Error from backend: %v", backendErr.Error())
	}
}

func main() {
	http.HandleFunc("/", printInfo)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
