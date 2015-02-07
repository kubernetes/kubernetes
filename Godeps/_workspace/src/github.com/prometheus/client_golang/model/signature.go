// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"bytes"
	"hash"
	"hash/fnv"
)

// SeparatorByte is a byte that cannot occur in valid UTF-8 sequences and is
// used to separate label names, label values, and other strings from each other
// when calculating their combined hash value (aka signature aka fingerprint).
const SeparatorByte byte = 255

var (
	// cache the signature of an empty label set.
	emptyLabelSignature = fnv.New64a().Sum64()

	hashAndBufPool = make(chan *hashAndBuf, 1024)
)

type hashAndBuf struct {
	h hash.Hash64
	b bytes.Buffer
}

func getHashAndBuf() *hashAndBuf {
	select {
	case hb := <-hashAndBufPool:
		return hb
	default:
		return &hashAndBuf{h: fnv.New64a()}
	}
}

func putHashAndBuf(hb *hashAndBuf) {
	select {
	case hashAndBufPool <- hb:
	default:
	}
}

// LabelsToSignature returns a unique signature (i.e., fingerprint) for a given
// label set.
func LabelsToSignature(labels map[string]string) uint64 {
	if len(labels) == 0 {
		return emptyLabelSignature
	}

	var result uint64
	hb := getHashAndBuf()
	defer putHashAndBuf(hb)

	for k, v := range labels {
		hb.b.WriteString(k)
		hb.b.WriteByte(SeparatorByte)
		hb.b.WriteString(v)
		hb.h.Write(hb.b.Bytes())
		result ^= hb.h.Sum64()
		hb.h.Reset()
		hb.b.Reset()
	}
	return result
}

// LabelValuesToSignature returns a unique signature (i.e., fingerprint) for the
// values of a given label set.
func LabelValuesToSignature(labels map[string]string) uint64 {
	if len(labels) == 0 {
		return emptyLabelSignature
	}

	var result uint64
	hb := getHashAndBuf()
	defer putHashAndBuf(hb)

	for _, v := range labels {
		hb.b.WriteString(v)
		hb.h.Write(hb.b.Bytes())
		result ^= hb.h.Sum64()
		hb.h.Reset()
		hb.b.Reset()
	}
	return result
}
