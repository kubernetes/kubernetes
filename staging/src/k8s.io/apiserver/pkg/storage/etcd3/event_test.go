/*
Copyright 2019 The Kubernetes Authors.

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

package etcd3

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"testing"
)

func TestParseEvent(t *testing.T) {
	for _, tc := range []struct {
		name          string
		etcdEvent     *clientv3.Event
		expectedEvent *event
		expectedErr   string
	}{
		{
			name: "successful create",
			etcdEvent: &clientv3.Event{
				Type:   clientv3.EventTypePut,
				PrevKv: nil,
				Kv: &mvccpb.KeyValue{
					// key is the key in bytes. An empty key is not allowed.
					Key:            []byte("key"),
					ModRevision:    1,
					CreateRevision: 1,
					Value:          []byte("value"),
				},
			},
			expectedEvent: &event{
				key:       "key",
				value:     []byte("value"),
				prevValue: nil,
				rev:       1,
				isDeleted: false,
				isCreated: true,
			},
			expectedErr: "",
		},
		{
			name: "unsuccessful delete",
			etcdEvent: &clientv3.Event{
				Type:   mvccpb.DELETE,
				PrevKv: nil,
				Kv: &mvccpb.KeyValue{
					Key:            []byte("key"),
					CreateRevision: 1,
					ModRevision:    2,
					Value:          nil,
				},
			},
			expectedErr: "etcd event received with PrevKv=nil",
		},
		{
			name: "successful delete",
			etcdEvent: &clientv3.Event{
				Type: mvccpb.DELETE,
				PrevKv: &mvccpb.KeyValue{
					Key:            []byte("key"),
					CreateRevision: 1,
					ModRevision:    1,
					Value:          []byte("value"),
				},
				Kv: &mvccpb.KeyValue{
					Key:            []byte("key"),
					CreateRevision: 1,
					ModRevision:    2,
					Value:          nil,
				},
			},
			expectedEvent: &event{
				key:       "key",
				value:     nil,
				prevValue: []byte("value"),
				rev:       2,
				isDeleted: true,
				isCreated: false,
			},
			expectedErr: "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			actualEvent, err := parseEvent(tc.etcdEvent)
			if tc.expectedErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expectedEvent, actualEvent)
			}
		})
	}
}
