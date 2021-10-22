// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package ocgrpc_test

import (
	"context"
	"io"
	"reflect"
	"testing"

	"go.opencensus.io/internal/testpb"
	"go.opencensus.io/plugin/ocgrpc"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

var keyAccountId = tag.MustNewKey("account_id")

func TestEndToEnd_Single(t *testing.T) {
	view.Register(ocgrpc.DefaultClientViews...)
	defer view.Unregister(ocgrpc.DefaultClientViews...)
	view.Register(ocgrpc.DefaultServerViews...)
	defer view.Unregister(ocgrpc.DefaultServerViews...)

	extraViews := []*view.View{
		ocgrpc.ServerReceivedMessagesPerRPCView,
		ocgrpc.ClientReceivedMessagesPerRPCView,
		ocgrpc.ServerSentMessagesPerRPCView,
		ocgrpc.ClientSentMessagesPerRPCView,
	}
	view.Register(extraViews...)
	defer view.Unregister(extraViews...)

	client, done := testpb.NewTestClient(t)
	defer done()

	ctx := context.Background()
	ctx, _ = tag.New(ctx, tag.Insert(keyAccountId, "abc123"))

	var (
		clientMethodTag        = tag.Tag{Key: ocgrpc.KeyClientMethod, Value: "testpb.Foo/Single"}
		serverMethodTag        = tag.Tag{Key: ocgrpc.KeyServerMethod, Value: "testpb.Foo/Single"}
		clientStatusOKTag      = tag.Tag{Key: ocgrpc.KeyClientStatus, Value: "OK"}
		serverStatusOKTag      = tag.Tag{Key: ocgrpc.KeyServerStatus, Value: "OK"}
		serverStatusUnknownTag = tag.Tag{Key: ocgrpc.KeyClientStatus, Value: "UNKNOWN"}
		clientStatusUnknownTag = tag.Tag{Key: ocgrpc.KeyServerStatus, Value: "UNKNOWN"}
	)

	_, err := client.Single(ctx, &testpb.FooRequest{})
	if err != nil {
		t.Fatal(err)
	}
	checkCount(t, ocgrpc.ClientCompletedRPCsView, 1, clientMethodTag, clientStatusOKTag)
	checkCount(t, ocgrpc.ServerCompletedRPCsView, 1, serverMethodTag, serverStatusOKTag)

	_, _ = client.Single(ctx, &testpb.FooRequest{Fail: true})
	checkCount(t, ocgrpc.ClientCompletedRPCsView, 1, clientMethodTag, serverStatusUnknownTag)
	checkCount(t, ocgrpc.ServerCompletedRPCsView, 1, serverMethodTag, clientStatusUnknownTag)

	tcs := []struct {
		v    *view.View
		tags []tag.Tag
		mean float64
	}{
		{ocgrpc.ClientSentMessagesPerRPCView, []tag.Tag{clientMethodTag}, 1.0},
		{ocgrpc.ServerReceivedMessagesPerRPCView, []tag.Tag{serverMethodTag}, 1.0},
		{ocgrpc.ClientReceivedMessagesPerRPCView, []tag.Tag{clientMethodTag}, 0.5},
		{ocgrpc.ServerSentMessagesPerRPCView, []tag.Tag{serverMethodTag}, 0.5},
		{ocgrpc.ClientSentBytesPerRPCView, []tag.Tag{clientMethodTag}, 1.0},
		{ocgrpc.ServerReceivedBytesPerRPCView, []tag.Tag{serverMethodTag}, 1.0},
		{ocgrpc.ClientReceivedBytesPerRPCView, []tag.Tag{clientMethodTag}, 0.0},
		{ocgrpc.ServerSentBytesPerRPCView, []tag.Tag{serverMethodTag}, 0.0},
	}

	for _, tt := range tcs {
		t.Run("view="+tt.v.Name, func(t *testing.T) {
			dist := getDistribution(t, tt.v, tt.tags...)
			if got, want := dist.Count, int64(2); got != want {
				t.Errorf("Count = %d; want %d", got, want)
			}
			if got, want := dist.Mean, tt.mean; got != want {
				t.Errorf("Mean = %v; want %v", got, want)
			}
		})
	}
}

func TestEndToEnd_Stream(t *testing.T) {
	view.Register(ocgrpc.DefaultClientViews...)
	defer view.Unregister(ocgrpc.DefaultClientViews...)
	view.Register(ocgrpc.DefaultServerViews...)
	defer view.Unregister(ocgrpc.DefaultServerViews...)

	extraViews := []*view.View{
		ocgrpc.ServerReceivedMessagesPerRPCView,
		ocgrpc.ClientReceivedMessagesPerRPCView,
		ocgrpc.ServerSentMessagesPerRPCView,
		ocgrpc.ClientSentMessagesPerRPCView,
	}
	view.Register(extraViews...)
	defer view.Unregister(extraViews...)

	client, done := testpb.NewTestClient(t)
	defer done()

	ctx := context.Background()
	ctx, _ = tag.New(ctx, tag.Insert(keyAccountId, "abc123"))

	var (
		clientMethodTag   = tag.Tag{Key: ocgrpc.KeyClientMethod, Value: "testpb.Foo/Multiple"}
		serverMethodTag   = tag.Tag{Key: ocgrpc.KeyServerMethod, Value: "testpb.Foo/Multiple"}
		clientStatusOKTag = tag.Tag{Key: ocgrpc.KeyClientStatus, Value: "OK"}
		serverStatusOKTag = tag.Tag{Key: ocgrpc.KeyServerStatus, Value: "OK"}
	)

	const msgCount = 3

	stream, err := client.Multiple(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < msgCount; i++ {
		stream.Send(&testpb.FooRequest{})
		_, err := stream.Recv()
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatal(err)
	}
	if _, err = stream.Recv(); err != io.EOF {
		t.Fatal(err)
	}

	checkCount(t, ocgrpc.ClientCompletedRPCsView, 1, clientMethodTag, clientStatusOKTag)
	checkCount(t, ocgrpc.ServerCompletedRPCsView, 1, serverMethodTag, serverStatusOKTag)

	tcs := []struct {
		v   *view.View
		tag tag.Tag
	}{
		{ocgrpc.ClientSentMessagesPerRPCView, clientMethodTag},
		{ocgrpc.ServerReceivedMessagesPerRPCView, serverMethodTag},
		{ocgrpc.ServerSentMessagesPerRPCView, serverMethodTag},
		{ocgrpc.ClientReceivedMessagesPerRPCView, clientMethodTag},
	}
	for _, tt := range tcs {
		serverSent := getDistribution(t, tt.v, tt.tag)
		if got, want := serverSent.Mean, float64(msgCount); got != want {
			t.Errorf("%q.Count = %v; want %v", ocgrpc.ServerSentMessagesPerRPCView.Name, got, want)
		}
	}
}

func checkCount(t *testing.T, v *view.View, want int64, tags ...tag.Tag) {
	if got, ok := getCount(t, v, tags...); ok && got != want {
		t.Errorf("View[name=%q].Row[tags=%v].Data = %d; want %d", v.Name, tags, got, want)
	}
}

func getCount(t *testing.T, v *view.View, tags ...tag.Tag) (int64, bool) {
	if len(tags) != len(v.TagKeys) {
		t.Errorf("Invalid tag specification, want %#v tags got %#v", v.TagKeys, tags)
		return 0, false
	}
	for i := range v.TagKeys {
		if tags[i].Key != v.TagKeys[i] {
			t.Errorf("Invalid tag specification, want %#v tags got %#v", v.TagKeys, tags)
			return 0, false
		}
	}
	rows, err := view.RetrieveData(v.Name)
	if err != nil {
		t.Fatal(err)
	}
	var foundRow *view.Row
	for _, row := range rows {
		if reflect.DeepEqual(row.Tags, tags) {
			foundRow = row
			break
		}
	}
	if foundRow == nil {
		var gotTags [][]tag.Tag
		for _, row := range rows {
			gotTags = append(gotTags, row.Tags)
		}
		t.Errorf("Failed to find row with keys %v among:\n%v", tags, gotTags)
		return 0, false
	}
	return foundRow.Data.(*view.CountData).Value, true
}

func getDistribution(t *testing.T, v *view.View, tags ...tag.Tag) *view.DistributionData {
	if len(tags) != len(v.TagKeys) {
		t.Fatalf("Invalid tag specification, want %#v tags got %#v", v.TagKeys, tags)
		return nil
	}
	for i := range v.TagKeys {
		if tags[i].Key != v.TagKeys[i] {
			t.Fatalf("Invalid tag specification, want %#v tags got %#v", v.TagKeys, tags)
			return nil
		}
	}
	rows, err := view.RetrieveData(v.Name)
	if err != nil {
		t.Fatal(err)
	}
	var foundRow *view.Row
	for _, row := range rows {
		if reflect.DeepEqual(row.Tags, tags) {
			foundRow = row
			break
		}
	}
	if foundRow == nil {
		var gotTags [][]tag.Tag
		for _, row := range rows {
			gotTags = append(gotTags, row.Tags)
		}
		t.Fatalf("Failed to find row with keys %v among:\n%v", tags, gotTags)
		return nil
	}
	return foundRow.Data.(*view.DistributionData)
}
