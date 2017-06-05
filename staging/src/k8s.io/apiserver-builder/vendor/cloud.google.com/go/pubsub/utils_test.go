// Copyright 2016 Google Inc. All Rights Reserved.
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

package pubsub

import (
	"time"

	"golang.org/x/net/context"
)

type modDeadlineCall struct {
	subName  string
	deadline time.Duration
	ackIDs   []string
}

type acknowledgeCall struct {
	subName string
	ackIDs  []string
}

type testService struct {
	service

	// The arguments of each call to modifyAckDealine are written to this channel.
	modDeadlineCalled chan modDeadlineCall

	// The arguments of each call to acknowledge are written to this channel.
	acknowledgeCalled chan acknowledgeCall
}

func (s *testService) modifyAckDeadline(ctx context.Context, subName string, deadline time.Duration, ackIDs []string) error {
	s.modDeadlineCalled <- modDeadlineCall{
		subName:  subName,
		deadline: deadline,
		ackIDs:   ackIDs,
	}
	return nil
}

func (s *testService) acknowledge(ctx context.Context, subName string, ackIDs []string) error {
	s.acknowledgeCalled <- acknowledgeCall{
		subName: subName,
		ackIDs:  ackIDs,
	}
	return nil
}

func (s *testService) splitAckIDs(ids []string) ([]string, []string) {
	return ids, nil
}
