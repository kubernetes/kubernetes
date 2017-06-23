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

// Package kms transforms values for storage at rest using a KMS provider
package kms

import (
	"bytes"
	"fmt"

	"k8s.io/apiserver/pkg/storage/value"
)

type kmsTransformer struct {
	kmsService value.KmsService
}

func NewKMSTransformer(kmsService value.KmsService) *kmsTransformer {
	return &kmsTransformer{kmsService}
}

func (t *kmsTransformer) TransformFromStorage(data []byte, context value.Context) ([]byte, bool, error) {
	// TODO(sakshams): Consider iterate over all transformers instead of searching for one. Check which one is better.
	slices := bytes.SplitN(data, []byte{':'}, 2)
	if len(slices) != 2 {
		return []byte{}, false, fmt.Errorf("invalid data for gkms transformer")
	}
	keyname := string(slices[0])

	transformer, err := t.kmsService.GetReadingTransformer(keyname)
	if err != nil {
		return []byte{}, false, err
	}

	return transformer.TransformFromStorage(data, context)
}

func (t *kmsTransformer) TransformToStorage(data []byte, context value.Context) ([]byte, error) {
	transformer, err := t.kmsService.GetWritingTransformer()
	if err != nil {
		return []byte{}, err
	}
	return transformer.TransformToStorage(data, context)
}
