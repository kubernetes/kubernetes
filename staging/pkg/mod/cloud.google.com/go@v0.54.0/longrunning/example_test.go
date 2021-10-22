// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package longrunning

import (
	"context"
	"fmt"
	"time"

	"github.com/golang/protobuf/ptypes"
	"github.com/golang/protobuf/ptypes/duration"
	"github.com/golang/protobuf/ptypes/timestamp"
	pb "google.golang.org/genproto/googleapis/longrunning"
)

func bestMomentInHistory() (*Operation, error) {
	t, err := time.Parse("2006-01-02 15:04:05.999999999 -0700 MST", "2009-11-10 23:00:00 +0000 UTC")
	if err != nil {
		return nil, err
	}
	resp, err := ptypes.TimestampProto(t)
	if err != nil {
		return nil, err
	}
	respAny, err := ptypes.MarshalAny(resp)
	if err != nil {
		return nil, err
	}
	metaAny, err := ptypes.MarshalAny(ptypes.DurationProto(1 * time.Hour))
	return &Operation{
		proto: &pb.Operation{
			Name:     "best-moment",
			Done:     true,
			Metadata: metaAny,
			Result: &pb.Operation_Response{
				Response: respAny,
			},
		},
	}, err
}

func ExampleOperation_Wait() {
	// Complex computation, might take a long time.
	op, err := bestMomentInHistory()
	if err != nil {
		// TODO: Handle err.
	}
	var ts timestamp.Timestamp
	err = op.Wait(context.TODO(), &ts)
	if err != nil && !op.Done() {
		fmt.Println("failed to fetch operation status", err)
	} else if err != nil && op.Done() {
		fmt.Println("operation completed with error", err)
	} else {
		fmt.Println(ptypes.TimestampString(&ts))
	}
	// Output:
	// 2009-11-10T23:00:00Z
}

func ExampleOperation_Metadata() {
	op, err := bestMomentInHistory()
	if err != nil {
		// TODO: Handle err.
	}

	// The operation might contain metadata.
	// In this example, the metadata contains the estimated length of time
	// the operation might take to complete.
	var meta duration.Duration
	if err := op.Metadata(&meta); err != nil {
		// TODO: Handle err.
	}
	d, err := ptypes.Duration(&meta)
	if err == ErrNoMetadata {
		fmt.Println("no metadata")
	} else if err != nil {
		// TODO: Handle err.
	} else {
		fmt.Println(d)
	}
	// Output:
	// 1h0m0s
}

func ExampleOperation_Cancel() {
	op, err := bestMomentInHistory()
	if err != nil {
		// TODO: Handle err.
	}
	if err := op.Cancel(context.Background()); err != nil {
		// TODO: Handle err.
	}
}

func ExampleOperation_Delete() {
	op, err := bestMomentInHistory()
	if err != nil {
		// TODO: Handle err.
	}
	if err := op.Delete(context.Background()); err != nil {
		// TODO: Handle err.
	}
}
