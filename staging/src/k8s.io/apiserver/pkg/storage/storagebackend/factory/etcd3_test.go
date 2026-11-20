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
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	noopoteltrace "go.opentelemetry.io/otel/trace/noop"

	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func TestClientCodec(t *testing.T) {
	server := testserver.RunEtcd(t)
	client, err := newETCD3Client(storagebackend.TransportConfig{
		ServerList:     server.Endpoints(),
		TracerProvider: noopoteltrace.NewTracerProvider(),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	key := "/codec/large"
	value := bytes.Repeat([]byte("x"), 1200*1024)
	watch := client.Watch(ctx, key)

	put, err := client.KV.Put(ctx, key, string(value))
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	rev := put.Header.Revision

	get, err := client.KV.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if len(get.Kvs) != 1 || !bytes.Equal(get.Kvs[0].Value, value) || get.Kvs[0].ModRevision != rev {
		t.Errorf("Get returned %d keys, want the value written at revision %d", len(get.Kvs), rev)
	}

	stream, err := client.KV.GetStream(ctx, key)
	if err != nil {
		t.Fatalf("GetStream failed: %v", err)
	}
	var streamed []byte
	for chunk := range stream {
		if err := chunk.Err(); err != nil {
			t.Fatalf("GetStream failed: %v", err)
		}
		for _, kv := range chunk.RangeResponse.Kvs {
			streamed = append(streamed, kv.Value...)
		}
	}
	if !bytes.Equal(streamed, value) {
		t.Errorf("GetStream returned %d bytes, want the %d written", len(streamed), len(value))
	}

	txn, err := client.KV.Txn(ctx).
		If(clientv3.Compare(clientv3.ModRevision(key), "=", 0)).
		Then(clientv3.OpPut(key, "unexpected")).
		Else(clientv3.OpGet(key)).
		Commit()
	if err != nil {
		t.Fatalf("Txn failed: %v", err)
	}
	if kvs := txn.Responses[0].GetResponseRange().Kvs; txn.Succeeded || len(kvs) != 1 || !bytes.Equal(kvs[0].Value, value) {
		t.Errorf("Txn succeeded=%v with %d keys, want a failed compare that returns the value", txn.Succeeded, len(kvs))
	}

	select {
	case resp := <-watch:
		if err := resp.Err(); err != nil {
			t.Fatalf("Watch failed: %v", err)
		}
		if len(resp.Events) != 1 || !bytes.Equal(resp.Events[0].Kv.Value, value) {
			t.Errorf("Watch returned %d events, want one with the value written", len(resp.Events))
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for the watch event")
	}
}

func TestEtcdCodecRejectsOtherTypes(t *testing.T) {
	if _, err := (etcdCodec{}).Marshal("not a message"); err == nil {
		t.Error("Marshal accepted a value that is not a proto.Message")
	}
	if err := (etcdCodec{}).Unmarshal(nil, "not a message"); err == nil {
		t.Error("Unmarshal accepted a value that is not a proto.Message")
	}
}

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
