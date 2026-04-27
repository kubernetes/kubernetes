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
	"errors"
	"io"
	"math/bits"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	errFieldCount          = errors.New("expected ListType to have 3 fields")
	errTypeMetaField       = errors.New("expected TypeMeta field to have TypeMeta type")
	errTypeMetaProtobufTag = errors.New(`expected TypeMeta protobuf field tag to be ""`)
	errListMetaField       = errors.New("expected ListMeta field to have ListMeta type")
	errListMetaProtobufTag = errors.New(`expected ListMeta protobuf field tag to be "bytes,1,opt,name=metadata"`)
	errItemsProtobufTag    = errors.New(`expected Items protobuf field tag to be "bytes,2,rep,name=items"`)
	errItemsSizer          = errors.New(`expected Items elements to implement proto.Sizer`)
)

// getStreamingListData implements list extraction logic for protobuf stream serialization.
//
// Reason for a custom logic instead of reusing accessors from meta package:
// * Validate proto tags to prevent incompatibility with proto standard package.
// * ListMetaAccessor doesn't distinguish empty from nil value.
// * TypeAccessor reparsing "apiVersion" and serializing it with "{group}/{version}"
func getStreamingListData(list runtime.Object) (data streamingListData, err error) {
	listValue, err := conversion.EnforcePtr(list)
	if err != nil {
		return data, err
	}
	listType := listValue.Type()
	if listType.NumField() != 3 {
		return data, errFieldCount
	}
	// TypeMeta: validated, but not returned as is not serialized.
	_, ok := listValue.Field(0).Interface().(metav1.TypeMeta)
	if !ok {
		return data, errTypeMetaField
	}
	if listType.Field(0).Tag.Get("protobuf") != "" {
		return data, errTypeMetaProtobufTag
	}
	// ListMeta
	listMeta, ok := listValue.Field(1).Interface().(metav1.ListMeta)
	if !ok {
		return data, errListMetaField
	}
	// if we were ever to relax the protobuf tag check we should update the hardcoded `0xa` below when writing ListMeta.
	if listType.Field(1).Tag.Get("protobuf") != "bytes,1,opt,name=metadata" {
		return data, errListMetaProtobufTag
	}
	data.listMeta = listMeta
	// Items; if we were ever to relax the protobuf tag check we should update the hardcoded `0x12` below when writing Items.
	if listType.Field(2).Tag.Get("protobuf") != "bytes,2,rep,name=items" {
		return data, errItemsProtobufTag
	}
	items, err := meta.ExtractList(list)
	if err != nil {
		return data, err
	}
	data.items = items
	data.totalSize, data.listMetaSize, data.itemsSizes, err = listSize(listMeta, items)
	return data, err
}

type streamingListData struct {
	// totalSize is the total size of the serialized List object, including their proto headers/size bytes
	totalSize int

	// listMetaSize caches results from .Size() call to listMeta, doesn't include header bytes (field identifier, size)
	listMetaSize int
	listMeta     metav1.ListMeta

	// itemsSizes caches results from .Size() call to items, doesn't include header bytes (field identifier, size)
	itemsSizes []int
	items      []runtime.Object
}

type sizer interface {
	Size() int
}

// listSize return size of ListMeta and items to be later used for preallocations.
// listMetaSize and itemSizes do not include header bytes (field identifier, size).
func listSize(listMeta metav1.ListMeta, items []runtime.Object) (totalSize, listMetaSize int, itemSizes []int, err error) {
	// ListMeta
	listMetaSize = listMeta.Size()
	totalSize += 1 + sovGenerated(uint64(listMetaSize)) + listMetaSize
	// Items
	itemSizes = make([]int, len(items))
	for i, item := range items {
		sizer, ok := item.(sizer)
		if !ok {
			return totalSize, listMetaSize, nil, errItemsSizer
		}
		n := sizer.Size()
		itemSizes[i] = n
		totalSize += 1 + sovGenerated(uint64(n)) + n
	}
	return totalSize, listMetaSize, itemSizes, nil
}

func streamingEncodeUnknownList(w io.Writer, unk runtime.Unknown, listData streamingListData, memAlloc runtime.MemoryAllocator) error {
	_, err := w.Write(protoEncodingPrefix)
	if err != nil {
		return err
	}
	// encodeList is responsible for encoding the List into the unknown Raw.
	encodeList := func(writer io.Writer) (int, error) {
		return streamingEncodeList(writer, listData, memAlloc)
	}
	_, err = unk.MarshalToWriter(w, listData.totalSize, encodeList)
	return err
}

func streamingEncodeList(w io.Writer, listData streamingListData, memAlloc runtime.MemoryAllocator) (size int, err error) {
	// ListMeta; 0xa = (1 << 3) | 2; field number: 1, type: 2 (LEN). https://protobuf.dev/programming-guides/encoding/#structure
	n, err := doEncodeWithHeader(&listData.listMeta, w, 0xa, listData.listMetaSize, memAlloc)
	size += n
	if err != nil {
		return size, err
	}
	// Items; 0x12 = (2 << 3) | 2; field number: 2, type: 2 (LEN). https://protobuf.dev/programming-guides/encoding/#structure
	for i, item := range listData.items {
		n, err := doEncodeWithHeader(item, w, 0x12, listData.itemsSizes[i], memAlloc)
		size += n
		if err != nil {
			return size, err
		}
	}
	return size, nil
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
