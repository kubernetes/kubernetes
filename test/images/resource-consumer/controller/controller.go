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
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	. "k8s.io/kubernetes/test/images/resource-consumer/common"
)

var port = flag.Int("port", 8080, "Port number.")
var consumerPort = flag.Int("consumer-port", 8080, "Port number of consumers.")
var consumerServiceName = flag.String("consumer-service-name", "resource-consumer", "Name of service containing resource consumers.")
var consumerServiceNamespace = flag.String("consumer-service-namespace", "default", "Namespace of service containing resource consumers.")

func main() {
	flag.Parse()
	mgr := NewController()
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), mgr))
}

type Controller struct {
	responseWriterLock sync.Mutex
	waitGroup          sync.WaitGroup
}

func NewController() *Controller {
	c := &Controller{}
	return c
}

func (handler *Controller) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(w, BadRequest, http.StatusBadRequest)
		return
	}
	// parsing POST request data and URL data
	if err := req.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// handle consumeCPU
	if req.URL.Path == ConsumeCPUAddress {
		handler.handleConsumeCPU(w, req.Form)
		return
	}
	// handle consumeMem
	if req.URL.Path == ConsumeMemAddress {
		handler.handleConsumeMem(w, req.Form)
		return
	}
	// handle bumpMetric
	if req.URL.Path == BumpMetricAddress {
		handler.handleBumpMetric(w, req.Form)
		return
	}
	http.Error(w, UnknownFunction, http.StatusNotFound)
}

func (handler *Controller) handleConsumeCPU(w http.ResponseWriter, query url.Values) {
	// getting string data for consumeCPU
	durationSecString := query.Get(DurationSecQuery)
	millicoresString := query.Get(MillicoresQuery)
	requestSizeInMillicoresString := query.Get(RequestSizeInMillicoresQuery)
	if durationSecString == "" || millicoresString == "" || requestSizeInMillicoresString == "" {
		http.Error(w, NotGivenFunctionArgument, http.StatusBadRequest)
		return
	}

	// convert data (strings to ints) for consumeCPU
	durationSec, durationSecError := strconv.Atoi(durationSecString)
	millicores, millicoresError := strconv.Atoi(millicoresString)
	requestSizeInMillicores, requestSizeInMillicoresError := strconv.Atoi(requestSizeInMillicoresString)
	if durationSecError != nil || millicoresError != nil || requestSizeInMillicoresError != nil || requestSizeInMillicores <= 0 {
		http.Error(w, IncorrectFunctionArgument, http.StatusBadRequest)
		return
	}

	count := millicores / requestSizeInMillicores
	rest := millicores - count*requestSizeInMillicores
	fmt.Fprintf(w, "RC manager: sending %v requests to consume %v millicores each and 1 request to consume %v millicores\n", count, requestSizeInMillicores, rest)
	if count > 0 {
		handler.waitGroup.Add(count)
		handler.sendConsumeCPURequests(w, count, requestSizeInMillicores, durationSec)
	}
	if rest > 0 {
		handler.waitGroup.Add(1)
		go handler.sendOneConsumeCPURequest(w, rest, durationSec)
	}
	handler.waitGroup.Wait()
}

func (handler *Controller) handleConsumeMem(w http.ResponseWriter, query url.Values) {
	// getting string data for consumeMem
	durationSecString := query.Get(DurationSecQuery)
	megabytesString := query.Get(MegabytesQuery)
	requestSizeInMegabytesString := query.Get(RequestSizeInMegabytesQuery)
	if durationSecString == "" || megabytesString == "" || requestSizeInMegabytesString == "" {
		http.Error(w, NotGivenFunctionArgument, http.StatusBadRequest)
		return
	}

	// convert data (strings to ints) for consumeMem
	durationSec, durationSecError := strconv.Atoi(durationSecString)
	megabytes, megabytesError := strconv.Atoi(megabytesString)
	requestSizeInMegabytes, requestSizeInMegabytesError := strconv.Atoi(requestSizeInMegabytesString)
	if durationSecError != nil || megabytesError != nil || requestSizeInMegabytesError != nil || requestSizeInMegabytes <= 0 {
		http.Error(w, IncorrectFunctionArgument, http.StatusBadRequest)
		return
	}

	count := megabytes / requestSizeInMegabytes
	rest := megabytes - count*requestSizeInMegabytes
	fmt.Fprintf(w, "RC manager: sending %v requests to consume %v MB each and 1 request to consume %v MB\n", count, requestSizeInMegabytes, rest)
	if count > 0 {
		handler.waitGroup.Add(count)
		handler.sendConsumeMemRequests(w, count, requestSizeInMegabytes, durationSec)
	}
	if rest > 0 {
		handler.waitGroup.Add(1)
		go handler.sendOneConsumeMemRequest(w, rest, durationSec)
	}
	handler.waitGroup.Wait()
}

func (handler *Controller) handleBumpMetric(w http.ResponseWriter, query url.Values) {
	// getting string data for handleBumpMetric
	metric := query.Get(MetricNameQuery)
	deltaString := query.Get(DeltaQuery)
	durationSecString := query.Get(DurationSecQuery)
	requestSizeCustomMetricString := query.Get(RequestSizeCustomMetricQuery)
	if durationSecString == "" || metric == "" || deltaString == "" || requestSizeCustomMetricString == "" {
		http.Error(w, NotGivenFunctionArgument, http.StatusBadRequest)
		return
	}

	// convert data (strings to ints/floats) for handleBumpMetric
	durationSec, durationSecError := strconv.Atoi(durationSecString)
	delta, deltaError := strconv.Atoi(deltaString)
	requestSizeCustomMetric, requestSizeCustomMetricError := strconv.Atoi(requestSizeCustomMetricString)
	if durationSecError != nil || deltaError != nil || requestSizeCustomMetricError != nil || requestSizeCustomMetric <= 0 {
		http.Error(w, IncorrectFunctionArgument, http.StatusBadRequest)
		return
	}

	count := delta / requestSizeCustomMetric
	rest := delta - count*requestSizeCustomMetric
	fmt.Fprintf(w, "RC manager: sending %v requests to bump custom metric by %v each and 1 request to bump by %v\n", count, requestSizeCustomMetric, rest)
	if count > 0 {
		handler.waitGroup.Add(count)
		handler.sendConsumeCustomMetric(w, metric, count, requestSizeCustomMetric, durationSec)
	}
	if rest > 0 {
		handler.waitGroup.Add(1)
		go handler.sendOneConsumeCustomMetric(w, metric, rest, durationSec)
	}
	handler.waitGroup.Wait()
}

func (manager *Controller) sendConsumeCPURequests(w http.ResponseWriter, requests, millicores, durationSec int) {
	for i := 0; i < requests; i++ {
		go manager.sendOneConsumeCPURequest(w, millicores, durationSec)
	}
}

func (manager *Controller) sendConsumeMemRequests(w http.ResponseWriter, requests, megabytes, durationSec int) {
	for i := 0; i < requests; i++ {
		go manager.sendOneConsumeMemRequest(w, megabytes, durationSec)
	}
}

func (manager *Controller) sendConsumeCustomMetric(w http.ResponseWriter, metric string, requests, delta, durationSec int) {
	for i := 0; i < requests; i++ {
		go manager.sendOneConsumeCustomMetric(w, metric, delta, durationSec)
	}
}

func createConsumerURL(suffix string) string {
	return fmt.Sprintf("http://%s.%s.svc.cluster.local:%d%s", *consumerServiceName, *consumerServiceNamespace, *consumerPort, suffix)
}

// sendOneConsumeCPURequest sends POST request for cpu consumption
func (c *Controller) sendOneConsumeCPURequest(w http.ResponseWriter, millicores int, durationSec int) {
	defer c.waitGroup.Done()
	query := createConsumerURL(ConsumeCPUAddress)
	_, err := http.PostForm(query, url.Values{MillicoresQuery: {strconv.Itoa(millicores)}, DurationSecQuery: {strconv.Itoa(durationSec)}})
	c.responseWriterLock.Lock()
	defer c.responseWriterLock.Unlock()
	if err != nil {
		fmt.Fprintf(w, "Failed to connect to consumer: %v\n", err)
		return
	}
	fmt.Fprintf(w, "Consumed %d millicores\n", millicores)
}

// sendOneConsumeMemRequest sends POST request for memory consumption
func (c *Controller) sendOneConsumeMemRequest(w http.ResponseWriter, megabytes int, durationSec int) {
	defer c.waitGroup.Done()
	query := createConsumerURL(ConsumeMemAddress)
	_, err := http.PostForm(query, url.Values{MegabytesQuery: {strconv.Itoa(megabytes)}, DurationSecQuery: {strconv.Itoa(durationSec)}})
	c.responseWriterLock.Lock()
	defer c.responseWriterLock.Unlock()
	if err != nil {
		fmt.Fprintf(w, "Failed to connect to consumer: %v\n", err)
		return
	}
	fmt.Fprintf(w, "Consumed %d megabytes\n", megabytes)
}

// sendOneConsumeCustomMetric sends POST request for custom metric consumption
func (c *Controller) sendOneConsumeCustomMetric(w http.ResponseWriter, customMetricName string, delta int, durationSec int) {
	defer c.waitGroup.Done()
	query := createConsumerURL(BumpMetricAddress)
	_, err := http.PostForm(query,
		url.Values{MetricNameQuery: {customMetricName}, DurationSecQuery: {strconv.Itoa(durationSec)}, DeltaQuery: {strconv.Itoa(delta)}})
	c.responseWriterLock.Lock()
	defer c.responseWriterLock.Unlock()
	if err != nil {
		fmt.Fprintf(w, "Failed to connect to consumer: %v\n", err)
		return
	}
	fmt.Fprintf(w, "Bumped metric %s by %d\n", customMetricName, delta)
}
