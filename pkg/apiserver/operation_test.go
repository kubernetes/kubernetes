/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	// TODO: remove dependency on api, apiserver should be generic
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
)

func TestOperation(t *testing.T) {
	ops := NewOperations()

	c := make(chan interface{})
	op := ops.NewOperation(c)
	// Allow context switch, so that op's ID can get added to the map and Get will work.
	// This is just so we can test Get. Ordinary users have no need to call Get immediately
	// after calling NewOperation, because it returns the operation directly.
	time.Sleep(time.Millisecond)
	go func() {
		time.Sleep(500 * time.Millisecond)
		c <- "All done"
	}()

	if op.expired(time.Now().Add(-time.Minute)) {
		t.Errorf("Expired before finished: %#v", op)
	}
	ops.expire(time.Minute)
	if tmp := ops.Get(op.ID); tmp == nil {
		t.Errorf("expire incorrectly removed the operation %#v", ops)
	}

	op.WaitFor(10 * time.Millisecond)
	if _, completed := op.StatusOrResult(); completed {
		t.Errorf("Unexpectedly fast completion")
	}

	const waiters = 10
	var waited int32
	for i := 0; i < waiters; i++ {
		go func() {
			op.WaitFor(time.Hour)
			atomic.AddInt32(&waited, 1)
		}()
	}

	op.WaitFor(time.Minute)
	if _, completed := op.StatusOrResult(); !completed {
		t.Errorf("Unexpectedly slow completion")
	}

	time.Sleep(100 * time.Millisecond)
	if waited != waiters {
		t.Errorf("Multiple waiters doesn't work, only %v finished", waited)
	}

	if op.expired(time.Now().Add(-time.Second)) {
		t.Errorf("Should not be expired: %#v", op)
	}
	if !op.expired(time.Now().Add(-80 * time.Millisecond)) {
		t.Errorf("Should be expired: %#v", op)
	}

	ops.expire(80 * time.Millisecond)
	if tmp := ops.Get(op.ID); tmp != nil {
		t.Errorf("expire failed to remove the operation %#v", ops)
	}

	if op.result.(string) != "All done" {
		t.Errorf("Got unexpected result: %#v", op.result)
	}
}

func TestOperationsList(t *testing.T) {
	testOver := make(chan struct{})
	defer close(testOver)
	simpleStorage := &SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			// Eliminate flakes by ensuring the create operation takes longer than this test.
			<-testOver
			return obj, nil
		},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": simpleStorage,
	}, codec, "/prefix/version")
	handler.(*defaultAPIServer).group.handler.asyncOpWait = 0
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{
		Name: "foo",
	}
	data, err := codec.Encode(simple)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	response, err := client.Post(server.URL+"/prefix/version/foo", "application/json", bytes.NewBuffer(data))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("Unexpected response %#v", response)
	}

	response, err = client.Get(server.URL + "/prefix/version/operations")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status code %#v", response)
	}
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := codec.Decode(body)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	oplist, ok := obj.(*api.ServerOpList)
	if !ok {
		t.Fatalf("expected ServerOpList, got %#v", obj)
	}
	if len(oplist.Items) != 1 {
		t.Errorf("expected 1 operation, got %#v", obj)
	}
}

func TestOpGet(t *testing.T) {
	testOver := make(chan struct{})
	defer close(testOver)
	simpleStorage := &SimpleRESTStorage{
		injectedFunction: func(obj interface{}) (interface{}, error) {
			// Eliminate flakes by ensuring the create operation takes longer than this test.
			<-testOver
			return obj, nil
		},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": simpleStorage,
	}, codec, "/prefix/version")
	handler.(*defaultAPIServer).group.handler.asyncOpWait = 0
	server := httptest.NewServer(handler)
	client := http.Client{}

	simple := Simple{
		Name: "foo",
	}
	data, err := codec.Encode(simple)
	t.Log(string(data))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	request, err := http.NewRequest("POST", server.URL+"/prefix/version/foo", bytes.NewBuffer(data))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("Unexpected response %#v", response)
	}

	var itemOut api.Status
	body, err := extractBody(response, &itemOut)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if itemOut.Status != api.StatusWorking || itemOut.Details == nil || itemOut.Details.ID == "" {
		t.Fatalf("Unexpected status: %#v (%s)", itemOut, string(body))
	}

	req2, err := http.NewRequest("GET", server.URL+"/prefix/version/operations/"+itemOut.Details.ID, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = client.Do(req2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if response.StatusCode != http.StatusAccepted {
		t.Errorf("Unexpected response %#v", response)
	}
}
