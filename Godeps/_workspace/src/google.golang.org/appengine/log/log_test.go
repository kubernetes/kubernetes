package log

import (
	"reflect"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"

	pb "google.golang.org/appengine/internal/log"
)

func TestQueryToRequest(t *testing.T) {
	testCases := []struct {
		desc  string
		query *Query
		want  *pb.LogReadRequest
	}{
		{
			desc:  "Empty",
			query: &Query{},
			want: &pb.LogReadRequest{
				AppId:     proto.String("s~fake"),
				VersionId: []string{"v12"},
			},
		},
		{
			desc: "Versions",
			query: &Query{
				Versions: []string{"alpha", "backend:beta"},
			},
			want: &pb.LogReadRequest{
				AppId: proto.String("s~fake"),
				ModuleVersion: []*pb.LogModuleVersion{
					{
						VersionId: proto.String("alpha"),
					}, {
						ModuleId:  proto.String("backend"),
						VersionId: proto.String("beta"),
					},
				},
			},
		},
	}

	for _, tt := range testCases {
		req, err := makeRequest(tt.query, "s~fake", "v12")

		if err != nil {
			t.Errorf("%s: got err %v, want nil", tt.desc, err)
			continue
		}
		if !proto.Equal(req, tt.want) {
			t.Errorf("%s request:\ngot  %v\nwant %v", tt.desc, req, tt.want)
		}
	}
}

func TestProtoToRecord(t *testing.T) {
	// We deliberately leave ModuleId and other optional fields unset.
	p := &pb.RequestLog{
		AppId:        proto.String("s~fake"),
		VersionId:    proto.String("1"),
		RequestId:    []byte("deadbeef"),
		Ip:           proto.String("127.0.0.1"),
		StartTime:    proto.Int64(431044244000000),
		EndTime:      proto.Int64(431044724000000),
		Latency:      proto.Int64(480000000),
		Mcycles:      proto.Int64(7),
		Method:       proto.String("GET"),
		Resource:     proto.String("/app"),
		HttpVersion:  proto.String("1.1"),
		Status:       proto.Int32(418),
		ResponseSize: proto.Int64(1337),
		UrlMapEntry:  proto.String("_go_app"),
		Combined:     proto.String("apache log"),
	}
	// Sanity check that all required fields are set.
	if _, err := proto.Marshal(p); err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	want := &Record{
		AppID:        "s~fake",
		ModuleID:     "default",
		VersionID:    "1",
		RequestID:    []byte("deadbeef"),
		IP:           "127.0.0.1",
		StartTime:    time.Date(1983, 8, 29, 22, 30, 44, 0, time.UTC),
		EndTime:      time.Date(1983, 8, 29, 22, 38, 44, 0, time.UTC),
		Latency:      8 * time.Minute,
		MCycles:      7,
		Method:       "GET",
		Resource:     "/app",
		HTTPVersion:  "1.1",
		Status:       418,
		ResponseSize: 1337,
		URLMapEntry:  "_go_app",
		Combined:     "apache log",
		Finished:     true,
		AppLogs:      []AppLog{},
	}
	got := protoToRecord(p)
	// Coerce locations to UTC since otherwise they will be in local.
	got.StartTime, got.EndTime = got.StartTime.UTC(), got.EndTime.UTC()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("protoToRecord:\ngot:  %v\nwant: %v", got, want)
	}
}
