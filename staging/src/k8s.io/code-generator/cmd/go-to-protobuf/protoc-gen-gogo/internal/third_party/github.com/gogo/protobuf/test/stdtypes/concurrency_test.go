package stdtypes

import (
	"io/ioutil"
	"sync"
	"testing"

	"k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo/internal/third_party/github.com/gogo/protobuf/proto"
)

func TestConcurrentTextMarshal(t *testing.T) {
	// Verify that there are no race conditions when calling
	// TextMarshaler.Marshal on a protobuf message that contains a StdDuration

	std := StdTypes{}
	var wg sync.WaitGroup

	tm := proto.TextMarshaler{}

	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := tm.Marshal(ioutil.Discard, &std)
			if err != nil {
				t.Fatal(err)
			}
		}()
	}
	wg.Wait()
}
