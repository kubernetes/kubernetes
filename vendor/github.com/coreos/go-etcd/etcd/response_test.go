package etcd

import (
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/ugorji/go/codec"
)

func createTestNode(size int) *Node {
	return &Node{
		Key:           strings.Repeat("a", 30),
		Value:         strings.Repeat("a", size),
		TTL:           123456789,
		ModifiedIndex: 123456,
		CreatedIndex:  123456,
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
		Action:    "aaaaa",
		Node:      createTestNodeWithChildren(children, size),
		PrevNode:  nil,
		EtcdIndex: 123456,
		RaftIndex: 123456,
		RaftTerm:  123456,
	}
}

func benchmarkResponseUnmarshalling(b *testing.B, children, size int) {
	response := createTestResponse(children, size)

	rr := RawResponse{http.StatusOK, make([]byte, 0), http.Header{}}
	codec.NewEncoderBytes(&rr.Body, new(codec.JsonHandle)).Encode(response)

	b.ResetTimer()
	newResponse := new(Response)
	var err error
	for i := 0; i < b.N; i++ {
		if newResponse, err = rr.Unmarshal(); err != nil {
			b.Errorf("Error: %v", err)
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
