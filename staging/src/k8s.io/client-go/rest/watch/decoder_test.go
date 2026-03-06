/*
Copyright 2014 The Kubernetes Authors.

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

package watch

import (
	"encoding/json"
	"fmt"
	"io"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/kubernetes/scheme"
)

// getDecoder mimics how k8s.io/client-go/rest.createSerializers creates a decoder
func getDecoder() runtime.Decoder {
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme.Scheme, scheme.Scheme, runtimejson.SerializerOptions{})
	directCodecFactory := scheme.Codecs.WithoutConversion()
	return directCodecFactory.DecoderToVersion(jsonSerializer, v1.SchemeGroupVersion)
}

func TestDecoder(t *testing.T) {
	tests := []struct {
		name                              string
		clientConcurrentWatchObjectDecode bool
	}{
		{
			name:                              "disable clientConcurrentWatchObjectDecode",
			clientConcurrentWatchObjectDecode: false,
		},
		{
			name:                              "enable clientConcurrentWatchObjectDecode",
			clientConcurrentWatchObjectDecode: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientConcurrentWatchObjectDecode, test.clientConcurrentWatchObjectDecode)

			table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error, watch.Bookmark}

			for _, eventType := range table {
				out, in := io.Pipe()

				decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())
				expect := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
				encoder := json.NewEncoder(in)
				eType := eventType
				errc := make(chan error)

				go func() {
					data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expect)
					if err != nil {
						errc <- fmt.Errorf("Unexpected error %v", err)
						return
					}
					event := metav1.WatchEvent{
						Type:   string(eType),
						Object: runtime.RawExtension{Raw: json.RawMessage(data)},
					}
					if err := encoder.Encode(&event); err != nil {
						t.Errorf("Unexpected error %v", err)
					}
					in.Close()
				}()

				done := make(chan struct{})
				go func() {
					action, got, err := decoder.Decode()
					if err != nil {
						errc <- fmt.Errorf("Unexpected error %v", err)
						return
					}
					if e, a := eType, action; e != a {
						t.Errorf("Expected %v, got %v", e, a)
					}
					if e, a := expect, got; !apiequality.Semantic.DeepDerivative(e, a) {
						t.Errorf("Expected %v, got %v", e, a)
					}
					t.Logf("Exited read")
					close(done)
				}()
				select {
				case err := <-errc:
					t.Fatal(err)
				case <-done:
				}

				done = make(chan struct{})
				go func() {
					_, _, err := decoder.Decode()
					if err == nil {
						t.Errorf("Unexpected nil error")
					}
					close(done)
				}()
				<-done

				decoder.Close()
			}
		})
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	tests := []struct {
		name                              string
		clientConcurrentWatchObjectDecode bool
	}{
		{
			name:                              "disable clientConcurrentWatchObjectDecode",
			clientConcurrentWatchObjectDecode: false,
		},
		{
			name:                              "enable clientConcurrentWatchObjectDecode",
			clientConcurrentWatchObjectDecode: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientConcurrentWatchObjectDecode, test.clientConcurrentWatchObjectDecode)
			out, in := io.Pipe()
			decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())

			done := make(chan struct{})

			go func() {
				_, _, err := decoder.Decode()
				if err == nil {
					t.Errorf("Unexpected nil error")
				}
				close(done)
			}()

			in.Close()

			select {
			case <-done:
				break
			case <-time.After(wait.ForeverTestTimeout):
				t.Error("Timeout")
			}
		})
	}
}

func TestDecoder_ConcurrentDecode(t *testing.T) {
	out, in := io.Pipe()
	d := &Decoder{
		decoder:         streaming.NewDecoder(out, getDecoder()),
		embeddedDecoder: getDecoder(),
	}
	encoder := json.NewEncoder(in)

	table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error, watch.Bookmark}
	eventList := []metav1.WatchEvent{}
	expect := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	for _, eventType := range table {
		data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expect)
		if err != nil {
			t.Fatalf("Unexpected error %v", err)
			return
		}
		eventList = append(eventList, metav1.WatchEvent{
			Type:   string(eventType),
			Object: runtime.RawExtension{Raw: json.RawMessage(data)},
		})
	}

	go func() {
		for _, event := range eventList {
			if err := encoder.Encode(&event); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
		}
		in.Close()
	}()

	resultChan := d.ConcurrentDecode()
	gotActions := []watch.EventType{}
	for {
		data, ok := <-resultChan
		if !ok {
			break
		}
		action := data.Action
		got := data.Object
		err := data.Err

		if err != nil {
			if err == io.EOF {
				break
			}
			t.Errorf("Unexpected error %v", err)
			return
		}
		gotActions = append(gotActions, action)
		if !apiequality.Semantic.DeepDerivative(expect, got) {
			t.Errorf("Expected %v, got %v", expect, got)
		}
	}

	if !apiequality.Semantic.DeepDerivative(table, gotActions) {
		t.Errorf("Expected %v, got %v", table, gotActions)
	}
}
