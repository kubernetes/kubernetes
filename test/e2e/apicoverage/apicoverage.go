/*
Copyright 2017 The Kubernetes Authors.

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
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/go-openapi/spec"
	"github.com/golang/glog"
)

var (
	openAPIFile = flag.String("openapi", "https://raw.githubusercontent.com/kubernetes/kubernetes/master/api/openapi-spec/swagger.json", "URL to openapi-spec of Kubernetes. If not specifying, the openapi-spec is download from https://raw.githubusercontent.com/kubernetes/kubernetes/master/api/openapi-spec/swagger.json instead")
	restLog     = flag.String("restlog", "", "File path to REST API operation log of Kubernetes")
	showAPIType = flag.String("apitype", "stable", "API type to show not-tested APIs. The options are stable, alpha, beta and all")
)

type apiData struct {
	Method string
	URL    string
}

type apiArray []apiData

var reOpenapi = regexp.MustCompile(`({\S+?})`)

func parseOpenAPI(openapi string) apiArray {
	var swaggerSpec spec.Swagger
	var apisOpenapi apiArray

	resp, err := http.Get(openapi)
	if err != nil {
		log.Fatal(err)
	}
	bytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	err = swaggerSpec.UnmarshalJSON(bytes)
	if err != nil {
		log.Fatal(err)
	}

	for path, pathItem := range swaggerSpec.Paths.Paths {
		// Standard HTTP methods: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#path-item-object
		methods := []string{"get", "put", "post", "delete", "options", "head", "patch"}
		for _, method := range methods {
			methodSpec, err := pathItem.JSONLookup(method)
			if err != nil {
				log.Fatal(err)
			}
			t, ok := methodSpec.(*spec.Operation)
			if ok == false {
				log.Fatal("Failed to convert methodSpec.")
			}
			if t == nil {
				continue
			}
			method := strings.ToUpper(method)
			api := apiData{
				Method: method,
				URL:    path,
			}
			apisOpenapi = append(apisOpenapi, api)
		}
	}
	return apisOpenapi
}

//   I0919 15:34:14.943642    6611 round_trippers.go:414] GET https://172.27.138.63:6443/api/v1/namespaces/kube-system/replicationcontrollers
var reAPILog = regexp.MustCompile(`round_trippers.go:\d+\] (GET|PUT|POST|DELETE|OPTIONS|HEAD|PATCH) (\S+)`)

func parseAPILog(restlog string) apiArray {
	var fp *os.File
	var apisLog apiArray
	var err error

	fp, err = os.Open(restlog)
	if err != nil {
		log.Fatal(err)
	}
	defer fp.Close()

	reader := bufio.NewReaderSize(fp, 4096)
	for line := ""; err == nil; line, err = reader.ReadString('\n') {
		result := reAPILog.FindSubmatch([]byte(line))
		if len(result) == 0 {
			continue
		}
		method := strings.ToUpper(string(result[1]))
		url := string(result[2])
		urlParts := strings.Split(url, "?")

		api := apiData{
			Method: method,
			URL:    urlParts[0],
		}
		apisLog = append(apisLog, api)
	}
	return apisLog
}

var reAlphaAPI = regexp.MustCompile(`\S+alpha\S+`)
var reBetaAPI = regexp.MustCompile(`\S+beta\S+`)

func main() {
	var found bool
	var apisTested apiArray
	var apisNotTested apiArray
	var apisNotTestedAlpha apiArray
	var apisNotTestedBeta apiArray
	var apisNotTestedStable apiArray

	flag.Parse()
	if len(*restLog) == 0 {
		glog.Fatal("need to set '--restlog'")
	}

	apisOpenapi := parseOpenAPI(*openAPIFile)
	apisLogs := parseAPILog(*restLog)

	for _, openapi := range apisOpenapi {
		regURL := reOpenapi.ReplaceAllLiteralString(openapi.URL, `[^/\s]+`) + `$`
		reg := regexp.MustCompile(regURL)
		found = false
		for _, log := range apisLogs {
			if openapi.Method != log.Method {
				continue
			}
			if !reg.MatchString(log.URL) {
				continue
			}
			found = true
			apisTested = append(apisTested, openapi)
			break
		}
		if found {
			continue
		}
		apisNotTested = append(apisNotTested, openapi)

		result := reAlphaAPI.FindSubmatch([]byte(openapi.URL))
		if len(result) != 0 {
			apisNotTestedAlpha = append(apisNotTestedAlpha, openapi)
			continue
		}
		result = reBetaAPI.FindSubmatch([]byte(openapi.URL))
		if len(result) != 0 {
			apisNotTestedBeta = append(apisNotTestedBeta, openapi)
			continue
		}
		apisNotTestedStable = append(apisNotTestedStable, openapi)
	}
	fmt.Printf("All APIs    : %d\n", len(apisOpenapi))
	fmt.Printf("numTested   : %d\n", len(apisTested))
	fmt.Printf("numNotTested: %d\n", len(apisNotTested))
	fmt.Printf("  numStableAPIs: %d\n", len(apisNotTestedStable))
	fmt.Printf("  numBetaAPIs  : %d\n", len(apisNotTestedBeta))
	fmt.Printf("  numAlphaAPIs : %d\n", len(apisNotTestedAlpha))

	if *showAPIType == "stable" || *showAPIType == "all" {
		fmt.Printf("Untested stable APIs:\n")
		for _, openapi := range apisNotTestedStable {
			fmt.Printf("  %s %s\n", openapi.Method, openapi.URL)
		}
	}
	if *showAPIType == "beta" || *showAPIType == "all" {
		fmt.Printf("Untested beta APIs:\n")
		for _, openapi := range apisNotTestedBeta {
			fmt.Printf("  %s %s\n", openapi.Method, openapi.URL)
		}
	}
	if *showAPIType == "alpha" || *showAPIType == "all" {
		fmt.Printf("Untested alpha APIs:\n")
		for _, openapi := range apisNotTestedAlpha {
			fmt.Printf("  %s %s\n", openapi.Method, openapi.URL)
		}
	}
}
