/*
 *
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package stream provides an interface for bidirectional streaming to the S2A server.
package stream

import (
	s2av2pb "github.com/google/s2a-go/internal/proto/v2/s2a_go_proto"
)

// S2AStream defines the operation for communicating with the S2A server over a bidirectional stream.
type S2AStream interface {
	// Send sends the message to the S2A server.
	Send(*s2av2pb.SessionReq) error
	// Recv receives the message from the S2A server.
	Recv() (*s2av2pb.SessionResp, error)
	// Closes the channel to the S2A server.
	CloseSend() error
}
