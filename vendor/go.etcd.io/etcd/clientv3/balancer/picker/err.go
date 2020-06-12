// Copyright 2018 The etcd Authors
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

package picker

import (
	"context"

	"google.golang.org/grpc/balancer"
)

// NewErr returns a picker that always returns err on "Pick".
func NewErr(err error) Picker {
	return &errPicker{p: Error, err: err}
}

type errPicker struct {
	p   Policy
	err error
}

func (ep *errPicker) String() string {
	return ep.p.String()
}

func (ep *errPicker) Pick(context.Context, balancer.PickInfo) (balancer.SubConn, func(balancer.DoneInfo), error) {
	return nil, nil, ep.err
}
