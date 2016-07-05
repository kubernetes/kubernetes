/*
Copyright 2015 The Kubernetes Authors.

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

package podutil

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io/ioutil"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

func Gzip(pods <-chan *api.Pod) ([]byte, error) {
	return gzipList(List(pods))
}

func gzipList(list *api.PodList) ([]byte, error) {
	raw, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), list)
	if err != nil {
		return nil, err
	}

	zipped := &bytes.Buffer{}
	zw := gzip.NewWriter(zipped)
	_, err = bytes.NewBuffer(raw).WriteTo(zw)
	if err != nil {
		return nil, err
	}

	err = zw.Close()
	if err != nil {
		return nil, err
	}

	return zipped.Bytes(), nil
}

func Gunzip(gzipped []byte) <-chan *api.Pod {
	return Stream(gunzipList(gzipped))
}

func gunzipList(gzipped []byte) (*api.PodList, error) {
	zr, err := gzip.NewReader(bytes.NewReader(gzipped))
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	raw, err := ioutil.ReadAll(zr)
	if err != nil {
		return nil, err
	}

	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(), raw)
	if err != nil {
		return nil, err
	}

	podlist, ok := obj.(*api.PodList)
	if !ok {
		return nil, fmt.Errorf("expected *api.PodList instead of %T", obj)
	}

	return podlist, nil
}
