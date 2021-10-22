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

// Package iam supports the resource-specific operations of Google Cloud
// IAM (Identity and Access Management) for the Google Cloud Libraries.
// See https://cloud.google.com/iam for more about IAM.
//
// Users of the Google Cloud Libraries will typically not use this package
// directly. Instead they will begin with some resource that supports IAM, like
// a pubsub topic, and call its IAM method to get a Handle for that resource.
package iam

import (
	"context"
	"fmt"
	"time"

	gax "github.com/googleapis/gax-go/v2"
	pb "google.golang.org/genproto/googleapis/iam/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// client abstracts the IAMPolicy API to allow multiple implementations.
type client interface {
	Get(ctx context.Context, resource string) (*pb.Policy, error)
	Set(ctx context.Context, resource string, p *pb.Policy) error
	Test(ctx context.Context, resource string, perms []string) ([]string, error)
	GetWithVersion(ctx context.Context, resource string, requestedPolicyVersion int32) (*pb.Policy, error)
}

// grpcClient implements client for the standard gRPC-based IAMPolicy service.
type grpcClient struct {
	c pb.IAMPolicyClient
}

var withRetry = gax.WithRetry(func() gax.Retryer {
	return gax.OnCodes([]codes.Code{
		codes.DeadlineExceeded,
		codes.Unavailable,
	}, gax.Backoff{
		Initial:    100 * time.Millisecond,
		Max:        60 * time.Second,
		Multiplier: 1.3,
	})
})

func (g *grpcClient) Get(ctx context.Context, resource string) (*pb.Policy, error) {
	return g.GetWithVersion(ctx, resource, 1)
}

func (g *grpcClient) GetWithVersion(ctx context.Context, resource string, requestedPolicyVersion int32) (*pb.Policy, error) {
	var proto *pb.Policy
	md := metadata.Pairs("x-goog-request-params", fmt.Sprintf("%s=%v", "resource", resource))
	ctx = insertMetadata(ctx, md)

	err := gax.Invoke(ctx, func(ctx context.Context, _ gax.CallSettings) error {
		var err error
		proto, err = g.c.GetIamPolicy(ctx, &pb.GetIamPolicyRequest{
			Resource: resource,
			Options: &pb.GetPolicyOptions{
				RequestedPolicyVersion: requestedPolicyVersion,
			},
		})
		return err
	}, withRetry)
	if err != nil {
		return nil, err
	}
	return proto, nil
}

func (g *grpcClient) Set(ctx context.Context, resource string, p *pb.Policy) error {
	md := metadata.Pairs("x-goog-request-params", fmt.Sprintf("%s=%v", "resource", resource))
	ctx = insertMetadata(ctx, md)

	return gax.Invoke(ctx, func(ctx context.Context, _ gax.CallSettings) error {
		_, err := g.c.SetIamPolicy(ctx, &pb.SetIamPolicyRequest{
			Resource: resource,
			Policy:   p,
		})
		return err
	}, withRetry)
}

func (g *grpcClient) Test(ctx context.Context, resource string, perms []string) ([]string, error) {
	var res *pb.TestIamPermissionsResponse
	md := metadata.Pairs("x-goog-request-params", fmt.Sprintf("%s=%v", "resource", resource))
	ctx = insertMetadata(ctx, md)

	err := gax.Invoke(ctx, func(ctx context.Context, _ gax.CallSettings) error {
		var err error
		res, err = g.c.TestIamPermissions(ctx, &pb.TestIamPermissionsRequest{
			Resource:    resource,
			Permissions: perms,
		})
		return err
	}, withRetry)
	if err != nil {
		return nil, err
	}
	return res.Permissions, nil
}

// A Handle provides IAM operations for a resource.
type Handle struct {
	c        client
	resource string
}

// A Handle3 provides IAM operations for a resource. It is similar to a Handle, but provides access to newer IAM features (e.g., conditions).
type Handle3 struct {
	c        client
	resource string
	version  int32
}

// InternalNewHandle is for use by the Google Cloud Libraries only.
//
// InternalNewHandle returns a Handle for resource.
// The conn parameter refers to a server that must support the IAMPolicy service.
func InternalNewHandle(conn grpc.ClientConnInterface, resource string) *Handle {
	return InternalNewHandleGRPCClient(pb.NewIAMPolicyClient(conn), resource)
}

// InternalNewHandleGRPCClient is for use by the Google Cloud Libraries only.
//
// InternalNewHandleClient returns a Handle for resource using the given
// grpc service that implements IAM as a mixin
func InternalNewHandleGRPCClient(c pb.IAMPolicyClient, resource string) *Handle {
	return InternalNewHandleClient(&grpcClient{c: c}, resource)
}

// InternalNewHandleClient is for use by the Google Cloud Libraries only.
//
// InternalNewHandleClient returns a Handle for resource using the given
// client implementation.
func InternalNewHandleClient(c client, resource string) *Handle {
	return &Handle{
		c:        c,
		resource: resource,
	}
}

// V3 returns a Handle3, which is like Handle except it sets
// requestedPolicyVersion to 3 when retrieving a policy and policy.version to 3
// when storing a policy.
func (h *Handle) V3() *Handle3 {
	return &Handle3{
		c:        h.c,
		resource: h.resource,
		version:  3,
	}
}

// Policy retrieves the IAM policy for the resource.
func (h *Handle) Policy(ctx context.Context) (*Policy, error) {
	proto, err := h.c.Get(ctx, h.resource)
	if err != nil {
		return nil, err
	}
	return &Policy{InternalProto: proto}, nil
}

// SetPolicy replaces the resource's current policy with the supplied Policy.
//
// If policy was created from a prior call to Get, then the modification will
// only succeed if the policy has not changed since the Get.
func (h *Handle) SetPolicy(ctx context.Context, policy *Policy) error {
	return h.c.Set(ctx, h.resource, policy.InternalProto)
}

// TestPermissions returns the subset of permissions that the caller has on the resource.
func (h *Handle) TestPermissions(ctx context.Context, permissions []string) ([]string, error) {
	return h.c.Test(ctx, h.resource, permissions)
}

// A RoleName is a name representing a collection of permissions.
type RoleName string

// Common role names.
const (
	Owner  RoleName = "roles/owner"
	Editor RoleName = "roles/editor"
	Viewer RoleName = "roles/viewer"
)

const (
	// AllUsers is a special member that denotes all users, even unauthenticated ones.
	AllUsers = "allUsers"

	// AllAuthenticatedUsers is a special member that denotes all authenticated users.
	AllAuthenticatedUsers = "allAuthenticatedUsers"
)

// A Policy is a list of Bindings representing roles
// granted to members.
//
// The zero Policy is a valid policy with no bindings.
type Policy struct {
	// TODO(jba): when type aliases are available, put Policy into an internal package
	// and provide an exported alias here.

	// This field is exported for use by the Google Cloud Libraries only.
	// It may become unexported in a future release.
	InternalProto *pb.Policy
}

// Members returns the list of members with the supplied role.
// The return value should not be modified. Use Add and Remove
// to modify the members of a role.
func (p *Policy) Members(r RoleName) []string {
	b := p.binding(r)
	if b == nil {
		return nil
	}
	return b.Members
}

// HasRole reports whether member has role r.
func (p *Policy) HasRole(member string, r RoleName) bool {
	return memberIndex(member, p.binding(r)) >= 0
}

// Add adds member member to role r if it is not already present.
// A new binding is created if there is no binding for the role.
func (p *Policy) Add(member string, r RoleName) {
	b := p.binding(r)
	if b == nil {
		if p.InternalProto == nil {
			p.InternalProto = &pb.Policy{}
		}
		p.InternalProto.Bindings = append(p.InternalProto.Bindings, &pb.Binding{
			Role:    string(r),
			Members: []string{member},
		})
		return
	}
	if memberIndex(member, b) < 0 {
		b.Members = append(b.Members, member)
		return
	}
}

// Remove removes member from role r if it is present.
func (p *Policy) Remove(member string, r RoleName) {
	bi := p.bindingIndex(r)
	if bi < 0 {
		return
	}
	bindings := p.InternalProto.Bindings
	b := bindings[bi]
	mi := memberIndex(member, b)
	if mi < 0 {
		return
	}
	// Order doesn't matter for bindings or members, so to remove, move the last item
	// into the removed spot and shrink the slice.
	if len(b.Members) == 1 {
		// Remove binding.
		last := len(bindings) - 1
		bindings[bi] = bindings[last]
		bindings[last] = nil
		p.InternalProto.Bindings = bindings[:last]
		return
	}
	// Remove member.
	// TODO(jba): worry about multiple copies of m?
	last := len(b.Members) - 1
	b.Members[mi] = b.Members[last]
	b.Members[last] = ""
	b.Members = b.Members[:last]
}

// Roles returns the names of all the roles that appear in the Policy.
func (p *Policy) Roles() []RoleName {
	if p.InternalProto == nil {
		return nil
	}
	var rns []RoleName
	for _, b := range p.InternalProto.Bindings {
		rns = append(rns, RoleName(b.Role))
	}
	return rns
}

// binding returns the Binding for the suppied role, or nil if there isn't one.
func (p *Policy) binding(r RoleName) *pb.Binding {
	i := p.bindingIndex(r)
	if i < 0 {
		return nil
	}
	return p.InternalProto.Bindings[i]
}

func (p *Policy) bindingIndex(r RoleName) int {
	if p.InternalProto == nil {
		return -1
	}
	for i, b := range p.InternalProto.Bindings {
		if b.Role == string(r) {
			return i
		}
	}
	return -1
}

// memberIndex returns the index of m in b's Members, or -1 if not found.
func memberIndex(m string, b *pb.Binding) int {
	if b == nil {
		return -1
	}
	for i, mm := range b.Members {
		if mm == m {
			return i
		}
	}
	return -1
}

// insertMetadata inserts metadata into the given context
func insertMetadata(ctx context.Context, mds ...metadata.MD) context.Context {
	out, _ := metadata.FromOutgoingContext(ctx)
	out = out.Copy()
	for _, md := range mds {
		for k, v := range md {
			out[k] = append(out[k], v...)
		}
	}
	return metadata.NewOutgoingContext(ctx, out)
}

// A Policy3 is a list of Bindings representing roles granted to members.
//
// The zero Policy3 is a valid policy with no bindings.
//
// It is similar to a Policy, except a Policy3 provides direct access to the
// list of Bindings.
//
// The policy version is always set to 3.
type Policy3 struct {
	etag     []byte
	Bindings []*pb.Binding
}

// Policy retrieves the IAM policy for the resource.
//
// requestedPolicyVersion is always set to 3.
func (h *Handle3) Policy(ctx context.Context) (*Policy3, error) {
	proto, err := h.c.GetWithVersion(ctx, h.resource, h.version)
	if err != nil {
		return nil, err
	}
	return &Policy3{
		Bindings: proto.Bindings,
		etag:     proto.Etag,
	}, nil
}

// SetPolicy replaces the resource's current policy with the supplied Policy.
//
// If policy was created from a prior call to Get, then the modification will
// only succeed if the policy has not changed since the Get.
func (h *Handle3) SetPolicy(ctx context.Context, policy *Policy3) error {
	return h.c.Set(ctx, h.resource, &pb.Policy{
		Bindings: policy.Bindings,
		Etag:     policy.etag,
		Version:  h.version,
	})
}

// TestPermissions returns the subset of permissions that the caller has on the resource.
func (h *Handle3) TestPermissions(ctx context.Context, permissions []string) ([]string, error) {
	return h.c.Test(ctx, h.resource, permissions)
}
