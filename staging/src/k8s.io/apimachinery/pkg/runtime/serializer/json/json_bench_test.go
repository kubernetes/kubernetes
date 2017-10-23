package json

import (
	"bytes"
	"encoding/json"
	"testing"

	jsoniter "github.com/json-iterator/go"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func BenchmarkJsonEncoding(b *testing.B) {
	obj := schema.GroupVersionKind{Kind: "foo", Group: "bar", Version: "baz"}
	encoder := json.NewEncoder(bytes.NewBuffer(nil))
	for i := 0; i < b.N; i++ {
		encoder.Encode(obj)
	}
}

func BenchmarkJsoniterEncoding(b *testing.B) {
	obj := schema.GroupVersionKind{Kind: "foo", Group: "bar", Version: "baz"}
	encoder := jsoniter.ConfigCompatibleWithStandardLibrary.NewEncoder(bytes.NewBuffer(nil))
	for i := 0; i < b.N; i++ {
		encoder.Encode(obj)
	}
}
