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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
	utiltesting "k8s.io/client-go/util/testing"
)

// getDecoder mimics how k8s.io/client-go/rest.createSerializers creates a decoder
func getDecoder() runtime.Decoder {
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme.Scheme, scheme.Scheme, runtimejson.SerializerOptions{})
	directCodecFactory := scheme.Codecs.WithoutConversion()
	return directCodecFactory.DecoderToVersion(jsonSerializer, v1.SchemeGroupVersion)
}

func TestDecoder(t *testing.T) {
	table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error, watch.Bookmark}

	for _, eventType := range table {
		t.Run(string(eventType), func(t *testing.T) {
			ctx := t.Context()
			out, in := io.Pipe()
			defer assertNoCloseError(t, in)
			decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())
			defer decoder.Close()
			expect := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
			encoder := json.NewEncoder(in)
			eType := eventType

			encodeErrCh := make(chan error)
			go func() {
				defer close(encodeErrCh)
				data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expect)
				if err != nil {
					encodeErrCh <- fmt.Errorf("encode error: %w", err)
					return
				}
				event := metav1.WatchEvent{
					Type:   string(eType),
					Object: runtime.RawExtension{Raw: json.RawMessage(data)},
				}
				if err := encoder.Encode(&event); err != nil {
					encodeErrCh <- fmt.Errorf("encode error: %w", err)
					return
				}
			}()

			decodeErrCh := make(chan error)
			go func() {
				defer close(decodeErrCh)
				action, got, err := decoder.Decode()
				if err != nil {
					decodeErrCh <- fmt.Errorf("decode error: %w", err)
					return
				}
				assert.Equal(t, eType, action)
				assert.Equal(t, expect, got)
			}()

			// Wait for encoder and decoder to return without error
			err := utiltesting.WaitForAllChannelsToCloseWithTimeout(ctx,
				wait.ForeverTestTimeout, encodeErrCh, decodeErrCh)
			require.NoError(t, err)

			// Close the input pipe, which should cause the decoder to error
			require.NoError(t, in.Close())

			// Wait for decoder EOF error
			decodeErrCh = make(chan error)
			go func() {
				defer close(decodeErrCh)
				_, _, err := decoder.Decode()
				if err != nil {
					decodeErrCh <- err
				}
			}()

			// Wait for decoder EOF error
			decodeErr, err := utiltesting.WaitForChannelEventWithTimeout(ctx, wait.ForeverTestTimeout, decodeErrCh)
			require.NoError(t, err)
			require.Equal(t, io.EOF, decodeErr)
		})
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	ctx := t.Context()
	out, in := io.Pipe()
	defer assertNoCloseError(t, in)
	decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())
	defer decoder.Close()

	errCh := make(chan error)
	go func() {
		defer close(errCh)
		_, _, err := decoder.Decode()
		if err != nil {
			errCh <- err
		}
	}()

	// Close the input pipe, which should cause the decoder to error
	require.NoError(t, in.Close())

	// Wait for decoder EOF error
	decodeErr, err := utiltesting.WaitForChannelEventWithTimeout(ctx, wait.ForeverTestTimeout, errCh)
	require.NoError(t, err)
	require.Equal(t, io.EOF, decodeErr)

	// Wait for errCh to close
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, errCh)
	require.NoError(t, err)
}

// assertNoCloseError asserts that closing the Closer doesn't error.
// Safe to call in a defer to ensure Closer.Close is called.
func assertNoCloseError(t *testing.T, c io.Closer) {
	assert.NoError(t, c.Close())
}
