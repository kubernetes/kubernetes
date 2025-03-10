/*
Copyright 2025 The Kubernetes Authors.

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

package protobuf

import (
	"fmt"
	"io"
	"math/bits"
	"reflect"

	"github.com/gogo/protobuf/proto"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// getListMeta implements list extraction logic for protobuf stream serialization.
//
// Reason for a custom logic instead of reusing accessors from meta package:
// * Validate proto tags to prevent incompatibility with proto standard package.
// * ListMetaAccessor doesn't distinguish empty from nil value.
// * TypeAccessor reparsing "apiVersion" and serializing it with "{group}/{version}"
func getListMeta(list runtime.Object) (metav1.TypeMeta, metav1.ListMeta, []runtime.Object, error) {
	listValue, err := conversion.EnforcePtr(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	listType := listValue.Type()
	if listType.NumField() != 3 {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListType to have 3 fields")
	}
	// TypeMeta
	typeMeta, ok := listValue.Field(0).Interface().(metav1.TypeMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field to have TypeMeta type")
	}
	if listType.Field(0).Tag.Get("protobuf") != "" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta protobuf field tag to be ""`)
	}
	// ListMeta
	listMeta, ok := listValue.Field(1).Interface().(metav1.ListMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field to have ListMeta type")
	}
	if listType.Field(1).Tag.Get("protobuf") != "bytes,1,opt,name=metadata" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected ListMeta protobuf field tag to be "bytes,1,opt,name=metadata"`)
	}
	// Items
	items, err := meta.ExtractList(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	if listType.Field(2).Tag.Get("protobuf") != "bytes,2,rep,name=items" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected Items protobuf field tag to be "bytes,2,rep,name=items"`)
	}
	for _, item := range items {
		if _, ok := item.(proto.Sizer); !ok {
			return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected Items elements to implement proto.Sizer`)
		}
	}
	return typeMeta, listMeta, items, nil
}

func streamingEncodeUnknownList(w io.Writer, unk runtime.Unknown, listMeta metav1.ListMeta, items []runtime.Object) error {
	_, err := w.Write(protoEncodingPrefix)
	if err != nil {
		return err
	}
	// TypeMeta
	_, err = encodeValue(w, []byte{0xa}, &unk.TypeMeta)
	if err != nil {
		return err
	}
	// Raw
	_, err = w.Write([]byte{0x12})
	if err != nil {
		return err
	}
	err = streamingEncodeList(w, listMeta, items)
	if err != nil {
		return err
	}
	// ContentEncoding
	_, err = encodeValue(w, []byte{0x1a}, unk.ContentEncoding)
	if err != nil {
		return err
	}
	// ContentType
	_, err = encodeValue(w, []byte{0x22}, unk.ContentType)
	if err != nil {
		return err
	}
	return nil
}

func streamingEncodeList(w io.Writer, listMeta metav1.ListMeta, items []runtime.Object) error {
	size, err := itemsSize(items)
	if err != nil {
		return err
	}
	size += embeddedSize(&listMeta)
	// ListMeta
	_, err = writeVarintGenerated(w, size)
	if err != nil {
		return err
	}
	listMetaSize, err := encodeValue(w, []byte{0xa}, &listMeta)
	if err != nil {
		return err
	}
	// Items
	listSize, err := streamingEncodeObjectSlice(w, items)
	if err != nil {
		return err
	}
	if listMetaSize+listSize != size {
		return fmt.Errorf("the size value was %d, but encoding wrote %d bytes to data", size, listMetaSize+listSize)
	}
	return nil
}

func streamingEncodeObjectSlice(w io.Writer, items []runtime.Object) (size int, err error) {
	for _, item := range items {
		n, err := encodeValue(w, []byte{0x12}, item)
		size += n
		if err != nil {
			return size, err
		}
	}
	return size, nil
}

func encodeValue(w io.Writer, field []byte, value any) (size int, err error) {
	size, err = w.Write(field)
	if err != nil {
		return size, err
	}
	switch v := value.(type) {
	case proto.Marshaler:
		data, err := v.Marshal()
		if err != nil {
			return size, err
		}
		n, err := writeVarintGenerated(w, len(data))
		if err != nil {
			return size, err
		}
		size += n
		n, err = w.Write(data)
		size += n
		return size, err
	case string:
		n, err := writeVarintGenerated(w, len(v))
		if err != nil {
			return size, err
		}
		size += n
		n, err = w.Write([]byte(v))
		size += n
		return size, err
	default:
		return size, errNotMarshalable{reflect.TypeOf(value)}
	}
}

func itemsSize(items []runtime.Object) (size int, err error) {
	for _, item := range items {
		sizer, ok := item.(proto.Sizer)
		if !ok {
			return size, fmt.Errorf("cannot stream Items elements that don't implement proto.Sizer")
		}
		size += embeddedSize(sizer)
	}
	return size, nil
}

func embeddedSize(sizer proto.Sizer) int {
	size := sizer.Size()
	// field number + varint + size
	return 1 + sovGenerated(uint64(size)) + size
}

func writeVarintGenerated(w io.Writer, v int) (int, error) {
	buf := make([]byte, sovGenerated(uint64(v)))
	encodeVarintGenerated(buf, len(buf), uint64(v))
	return w.Write(buf)
}

// sovGenerated is copied from `generated.pb.go` returns size of varint.
func sovGenerated(v uint64) int {
	return (bits.Len64(v|1) + 6) / 7
}

// encodeVarintGenerated is copied from `generated.pb.go` encodes varint.
func encodeVarintGenerated(dAtA []byte, offset int, v uint64) int {
	offset -= sovGenerated(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
