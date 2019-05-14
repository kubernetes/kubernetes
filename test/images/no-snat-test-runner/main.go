/*
Copyright 2019 The Kubernetes Authors.

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
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
)

const testPodPort = 8080

func doTest(url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	if code := resp.StatusCode; code >= 400 {
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		return fmt.Errorf("code=%d,url=%q,body=%q", code, url, string(body))
	}
	return nil
}

func main() {
	podIPs, ok := os.LookupEnv("POD_IPS")
	if !ok {
		log.Fatalf("Missing POD_IPS in env")
	}
	ips := strings.Split(podIPs, ":")
	for i, ip := range ips {
		ips[i] = fmt.Sprintf("%s:%d", ip, testPodPort)
	}
	errs := []string{}
	for i, ip := range ips {
		testIPs := strings.Join(append(ips[:i], ips[i+1:]...), ",")
		url := fmt.Sprintf("http://%s/checknosnat?ips=%s", ip, testIPs)
		if err := doTest(url); err != nil {
			errs = append(errs, fmt.Sprintf("%s\t%s", url, err.Error()))
		}
	}
	if len(errs) > 0 {
		fmt.Printf("Fail\n%s", strings.Join(errs, "\n"))
	} else {
		fmt.Println("Pass")
	}
}
