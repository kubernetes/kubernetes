/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package mock_routeguide_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/golang/protobuf/proto"
	rgmock "google.golang.org/grpc/examples/route_guide/mock_routeguide"
	rgpb "google.golang.org/grpc/examples/route_guide/routeguide"
)

var msg = &rgpb.RouteNote{
	Location: &rgpb.Point{Latitude: 17, Longitude: 29},
	Message:  "Taxi-cab",
}

func TestRouteChat(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create mock for the stream returned by RouteChat
	stream := rgmock.NewMockRouteGuide_RouteChatClient(ctrl)
	// set expectation on sending.
	stream.EXPECT().Send(
		gomock.Any(),
	).Return(nil)
	// Set expectation on receiving.
	stream.EXPECT().Recv().Return(msg, nil)
	stream.EXPECT().CloseSend().Return(nil)
	// Create mock for the client interface.
	rgclient := rgmock.NewMockRouteGuideClient(ctrl)
	// Set expectation on RouteChat
	rgclient.EXPECT().RouteChat(
		gomock.Any(),
	).Return(stream, nil)
	if err := testRouteChat(rgclient); err != nil {
		t.Fatalf("Test failed: %v", err)
	}
}

func testRouteChat(client rgpb.RouteGuideClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stream, err := client.RouteChat(ctx)
	if err != nil {
		return err
	}
	if err := stream.Send(msg); err != nil {
		return err
	}
	if err := stream.CloseSend(); err != nil {
		return err
	}
	got, err := stream.Recv()
	if err != nil {
		return err
	}
	if !proto.Equal(got, msg) {
		return fmt.Errorf("stream.Recv() = %v, want %v", got, msg)
	}
	return nil
}
