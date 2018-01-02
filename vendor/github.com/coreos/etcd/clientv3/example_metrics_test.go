// Copyright 2016 The etcd Authors
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

package clientv3_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
)

func ExampleMetrics_All() {
	// listen for all prometheus metrics
	go func() {
		http.Handle("/metrics", prometheus.Handler())
		log.Fatal(http.ListenAndServe(":47989", nil))
	}()

	url := "http://localhost:47989/metrics"

	// make an http request to fetch all prometheus metrics
	resp, err := http.Get(url)
	if err != nil {
		log.Fatalf("fetch error: %v", err)
	}
	b, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		log.Fatalf("fetch error: reading %s: %v", url, err)
	}
	fmt.Printf("%s", b)
}
