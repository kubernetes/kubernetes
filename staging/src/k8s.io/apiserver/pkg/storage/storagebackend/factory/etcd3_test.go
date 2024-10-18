/*
Copyright 2022 The Kubernetes Authors.

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

package factory

import (
	"context"
	"errors"
	"fmt"
	"k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"testing"
	"time"
)

func Test_atomicLastError(t *testing.T) {
	aError := &atomicLastError{err: fmt.Errorf("initial error")}
	// no timestamp is always updated
	aError.Store(errors.New("updated error"), time.Time{})
	err := aError.Load()
	if err.Error() != "updated error" {
		t.Fatalf("Expected: \"updated error\" got: %s", err.Error())
	}
	// update to current time
	now := time.Now()
	aError.Store(errors.New("now error"), now)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
	// no update to past time
	past := now.Add(-5 * time.Second)
	aError.Store(errors.New("past error"), past)
	err = aError.Load()
	if err.Error() != "now error" {
		t.Fatalf("Expected: \"now error\" got: %s", err.Error())
	}
}

func TestClientWithGrpcHealthcheck(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)

	etcdConfig := testserver.NewTestConfig(t)

	client := testserver.RunEtcd(t, etcdConfig)
	cfg := storagebackend.Config{
		Type: storagebackend.StorageTypeETCD3,
		Transport: storagebackend.TransportConfig{
			ServerList:            client.Endpoints(),
			EnableGrpcHealthcheck: true,
		},
		Codec: codec,
	}
	storage, destroyFunc, err := newETCD3Storage(*cfg.ForResource(schema.GroupResource{Resource: "pods"}), nil, nil, "")
	defer destroyFunc()
	if err != nil {
		t.Fatal(err)
	}
	err = storage.Create(context.TODO(), "/abc", &example.Pod{}, nil, 0)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
}
