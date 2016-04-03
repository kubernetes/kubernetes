package etcd

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMemberCollectionUnmarshal(t *testing.T) {
	tests := []struct {
		body []byte
		want memberCollection
	}{
		{
			body: []byte(`{"members":[]}`),
			want: memberCollection([]Member{}),
		},
		{
			body: []byte(`{"members":[{"id":"2745e2525fce8fe","peerURLs":["http://127.0.0.1:7003"],"name":"node3","clientURLs":["http://127.0.0.1:4003"]},{"id":"42134f434382925","peerURLs":["http://127.0.0.1:2380","http://127.0.0.1:7001"],"name":"node1","clientURLs":["http://127.0.0.1:2379","http://127.0.0.1:4001"]},{"id":"94088180e21eb87b","peerURLs":["http://127.0.0.1:7002"],"name":"node2","clientURLs":["http://127.0.0.1:4002"]}]}`),
			want: memberCollection(
				[]Member{
					{
						ID:   "2745e2525fce8fe",
						Name: "node3",
						PeerURLs: []string{
							"http://127.0.0.1:7003",
						},
						ClientURLs: []string{
							"http://127.0.0.1:4003",
						},
					},
					{
						ID:   "42134f434382925",
						Name: "node1",
						PeerURLs: []string{
							"http://127.0.0.1:2380",
							"http://127.0.0.1:7001",
						},
						ClientURLs: []string{
							"http://127.0.0.1:2379",
							"http://127.0.0.1:4001",
						},
					},
					{
						ID:   "94088180e21eb87b",
						Name: "node2",
						PeerURLs: []string{
							"http://127.0.0.1:7002",
						},
						ClientURLs: []string{
							"http://127.0.0.1:4002",
						},
					},
				},
			),
		},
	}

	for i, tt := range tests {
		var got memberCollection
		err := json.Unmarshal(tt.body, &got)
		if err != nil {
			t.Errorf("#%d: unexpected error: %v", i, err)
			continue
		}

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("#%d: incorrect output: want=%#v, got=%#v", i, tt.want, got)
		}
	}
}
