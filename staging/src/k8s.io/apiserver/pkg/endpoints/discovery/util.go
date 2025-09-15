/*
Copyright 2017 The Kubernetes Authors.

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

package discovery

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

const APIGroupPrefix = "/apis"

func keepUnversioned(group string) bool {
	return group == "" || group == "extensions"
}

// stripVersionEncoder strips APIVersion field from the encoding output. It's
// used to keep the responses at the discovery endpoints backward compatible
// with release-1.1, when the responses have empty APIVersion.
type stripVersionEncoder struct {
	encoder    runtime.Encoder
	serializer runtime.Serializer
	identifier runtime.Identifier
}

func newStripVersionEncoder(e runtime.Encoder, s runtime.Serializer) runtime.Encoder {
	return stripVersionEncoder{
		encoder:    e,
		serializer: s,
		identifier: identifier(e),
	}
}

func identifier(e runtime.Encoder) runtime.Identifier {
	result := map[string]string{
		"name": "stripVersion",
	}
	if e != nil {
		result["encoder"] = string(e.Identifier())
	}
	identifier, err := json.Marshal(result)
	if err != nil {
		klog.Fatalf("Failed marshaling identifier for stripVersionEncoder: %v", err)
	}
	return runtime.Identifier(identifier)
}

func (c stripVersionEncoder) Encode(obj runtime.Object, w io.Writer) error {
	if co, ok := obj.(runtime.CacheableObject); ok {
		return co.CacheEncode(c.Identifier(), c.doEncode, w)
	}
	return c.doEncode(obj, w)
}

func (c stripVersionEncoder) doEncode(obj runtime.Object, w io.Writer) error {
	buf := bytes.NewBuffer([]byte{})
	err := c.encoder.Encode(obj, buf)
	if err != nil {
		return err
	}
	roundTrippedObj, gvk, err := c.serializer.Decode(buf.Bytes(), nil, nil)
	if err != nil {
		return err
	}
	gvk.Group = ""
	gvk.Version = ""
	roundTrippedObj.GetObjectKind().SetGroupVersionKind(*gvk)
	return c.serializer.Encode(roundTrippedObj, w)
}

// Identifier implements runtime.Encoder interface.
func (c stripVersionEncoder) Identifier() runtime.Identifier {
	return c.identifier
}

// stripVersionNegotiatedSerializer will return stripVersionEncoder when
// EncoderForVersion is called. See comments for stripVersionEncoder.
type stripVersionNegotiatedSerializer struct {
	runtime.NegotiatedSerializer
}

func (n stripVersionNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	serializer, ok := encoder.(runtime.Serializer)
	if !ok {
		// The stripVersionEncoder needs both an encoder and decoder, but is called from a context that doesn't have access to the
		// decoder. We do a best effort cast here (since this code path is only for backwards compatibility) to get access to the caller's
		// decoder.
		panic(fmt.Sprintf("Unable to extract serializer from %#v", encoder))
	}
	versioned := n.NegotiatedSerializer.EncoderForVersion(encoder, gv)
	return newStripVersionEncoder(versioned, serializer)
}
