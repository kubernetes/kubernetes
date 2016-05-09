// Copyright 2015 CoreOS, Inc.
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

package rpctypes

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

var (
	ErrEmptyKey     = grpc.Errorf(codes.InvalidArgument, "etcdserver: key is not provided")
	ErrTooManyOps   = grpc.Errorf(codes.InvalidArgument, "etcdserver: too many operations in txn request")
	ErrDuplicateKey = grpc.Errorf(codes.InvalidArgument, "etcdserver: duplicate key given in txn request")
	ErrCompacted    = grpc.Errorf(codes.OutOfRange, "etcdserver: storage: required revision has been compacted")
	ErrFutureRev    = grpc.Errorf(codes.OutOfRange, "etcdserver: storage: required revision is a future revision")
	ErrNoSpace      = grpc.Errorf(codes.ResourceExhausted, "etcdserver: storage: database space exceeded")

	ErrLeaseNotFound = grpc.Errorf(codes.NotFound, "etcdserver: requested lease not found")
	ErrLeaseExist    = grpc.Errorf(codes.FailedPrecondition, "etcdserver: lease already exists")

	ErrMemberExist    = grpc.Errorf(codes.FailedPrecondition, "etcdserver: member ID already exist")
	ErrPeerURLExist   = grpc.Errorf(codes.FailedPrecondition, "etcdserver: Peer URLs already exists")
	ErrMemberBadURLs  = grpc.Errorf(codes.InvalidArgument, "etcdserver: given member URLs are invalid")
	ErrMemberNotFound = grpc.Errorf(codes.NotFound, "etcdserver: member not found")

	ErrRequestTooLarge = grpc.Errorf(codes.InvalidArgument, "etcdserver: request is too large")

	ErrUserAlreadyExist = grpc.Errorf(codes.FailedPrecondition, "etcdserver: user name already exists")
	ErrUserNotFound     = grpc.Errorf(codes.FailedPrecondition, "etcdserver: user name not found")
	ErrRoleAlreadyExist = grpc.Errorf(codes.FailedPrecondition, "etcdserver: role name already exists")
)
