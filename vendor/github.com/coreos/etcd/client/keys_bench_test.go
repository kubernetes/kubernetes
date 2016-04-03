// Copyright 2015 CoreOS, Inc.
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

package client

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func createTestNode(size int) *Node {
	return &Node{
		Key:           strings.Repeat("a", 30),
		Value:         strings.Repeat("a", size),
		CreatedIndex:  123456,
		ModifiedIndex: 123456,
		TTL:           123456789,
	}
}

func createTestNodeWithChildren(children, size int) *Node {
	node := createTestNode(size)
	for i := 0; i < children; i++ {
		node.Nodes = append(node.Nodes, createTestNode(size))
	}
	return node
}

func createTestResponse(children, size int) *Response {
	return &Response{
		Action:   "aaaaa",
		Node:     createTestNodeWithChildren(children, size),
		PrevNode: nil,
	}
}

func benchmarkResponseUnmarshalling(b *testing.B, children, size int) {
	header := http.Header{}
	header.Add("X-Etcd-Index", "123456")
	response := createTestResponse(children, size)
	body, err := json.Marshal(response)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	newResponse := new(Response)
	for i := 0; i < b.N; i++ {
		if newResponse, err = unmarshalSuccessfulKeysResponse(header, body); err != nil {
			b.Errorf("error unmarshaling response (%v)", err)
		}

	}
	if !reflect.DeepEqual(response.Node, newResponse.Node) {
		b.Errorf("Unexpected difference in a parsed response: \n%+v\n%+v", response, newResponse)
	}
}

func BenchmarkSmallResponseUnmarshal(b *testing.B) {
	benchmarkResponseUnmarshalling(b, 30, 20)
}

func BenchmarkManySmallResponseUnmarshal(b *testing.B) {
	benchmarkResponseUnmarshalling(b, 3000, 20)
}

func BenchmarkMediumResponseUnmarshal(b *testing.B) {
	benchmarkResponseUnmarshalling(b, 300, 200)
}

func BenchmarkLargeResponseUnmarshal(b *testing.B) {
	benchmarkResponseUnmarshalling(b, 3000, 2000)
}
