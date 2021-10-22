// Copyright 2017 Google LLC
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

package rpcreplay_test

import (
	"cloud.google.com/go/rpcreplay"
	"google.golang.org/grpc"
)

var serverAddress string

func Example_NewRecorder() {
	rec, err := rpcreplay.NewRecorder("service.replay", nil)
	if err != nil {
		// TODO: Handle error.
	}
	defer func() {
		if err := rec.Close(); err != nil {
			// TODO: Handle error.
		}
	}()
	conn, err := grpc.Dial(serverAddress, rec.DialOptions()...)
	if err != nil {
		// TODO: Handle error.
	}
	_ = conn // TODO: use connection
}

func Example_NewReplayer() {
	rep, err := rpcreplay.NewReplayer("service.replay")
	if err != nil {
		// TODO: Handle error.
	}
	defer rep.Close()
	conn, err := grpc.Dial(serverAddress, rep.DialOptions()...)
	if err != nil {
		// TODO: Handle error.
	}
	_ = conn // TODO: use connection
}
