// Copyright 2014 Google Inc. All Rights Reserved.
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

package storage

import (
	"fmt"

	"golang.org/x/net/context"
	raw "google.golang.org/api/storage/v1"
)

// ACLRole is the level of access to grant.
type ACLRole string

const (
	RoleOwner  ACLRole = "OWNER"
	RoleReader ACLRole = "READER"
)

// ACLEntity refers to a user or group.
// They are sometimes referred to as grantees.
//
// It could be in the form of:
// "user-<userId>", "user-<email>", "group-<groupId>", "group-<email>",
// "domain-<domain>" and "project-team-<projectId>".
//
// Or one of the predefined constants: AllUsers, AllAuthenticatedUsers.
type ACLEntity string

const (
	AllUsers              ACLEntity = "allUsers"
	AllAuthenticatedUsers ACLEntity = "allAuthenticatedUsers"
)

// ACLRule represents a grant for a role to an entity (user, group or team) for a Google Cloud Storage object or bucket.
type ACLRule struct {
	Entity ACLEntity
	Role   ACLRole
}

// ACLHandle provides operations on an access control list for a Google Cloud Storage bucket or object.
type ACLHandle struct {
	c         *Client
	bucket    string
	object    string
	isDefault bool
}

// Delete permanently deletes the ACL entry for the given entity.
func (a *ACLHandle) Delete(ctx context.Context, entity ACLEntity) error {
	if a.object != "" {
		return a.objectDelete(ctx, entity)
	}
	if a.isDefault {
		return a.bucketDefaultDelete(ctx, entity)
	}
	return a.bucketDelete(ctx, entity)
}

// Set sets the permission level for the given entity.
func (a *ACLHandle) Set(ctx context.Context, entity ACLEntity, role ACLRole) error {
	if a.object != "" {
		return a.objectSet(ctx, entity, role)
	}
	if a.isDefault {
		return a.bucketDefaultSet(ctx, entity, role)
	}
	return a.bucketSet(ctx, entity, role)
}

// List retrieves ACL entries.
func (a *ACLHandle) List(ctx context.Context) ([]ACLRule, error) {
	if a.object != "" {
		return a.objectList(ctx)
	}
	if a.isDefault {
		return a.bucketDefaultList(ctx)
	}
	return a.bucketList(ctx)
}

func (a *ACLHandle) bucketDefaultList(ctx context.Context) ([]ACLRule, error) {
	acls, err := a.c.raw.DefaultObjectAccessControls.List(a.bucket).Context(ctx).Do()
	if err != nil {
		return nil, fmt.Errorf("storage: error listing default object ACL for bucket %q: %v", a.bucket, err)
	}
	return toACLRules(acls.Items), nil
}

func (a *ACLHandle) bucketDefaultSet(ctx context.Context, entity ACLEntity, role ACLRole) error {
	acl := &raw.ObjectAccessControl{
		Bucket: a.bucket,
		Entity: string(entity),
		Role:   string(role),
	}
	_, err := a.c.raw.DefaultObjectAccessControls.Update(a.bucket, string(entity), acl).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error updating default ACL entry for bucket %q, entity %q: %v", a.bucket, entity, err)
	}
	return nil
}

func (a *ACLHandle) bucketDefaultDelete(ctx context.Context, entity ACLEntity) error {
	err := a.c.raw.DefaultObjectAccessControls.Delete(a.bucket, string(entity)).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error deleting default ACL entry for bucket %q, entity %q: %v", a.bucket, entity, err)
	}
	return nil
}

func (a *ACLHandle) bucketList(ctx context.Context) ([]ACLRule, error) {
	acls, err := a.c.raw.BucketAccessControls.List(a.bucket).Context(ctx).Do()
	if err != nil {
		return nil, fmt.Errorf("storage: error listing bucket ACL for bucket %q: %v", a.bucket, err)
	}
	r := make([]ACLRule, len(acls.Items))
	for i, v := range acls.Items {
		r[i].Entity = ACLEntity(v.Entity)
		r[i].Role = ACLRole(v.Role)
	}
	return r, nil
}

func (a *ACLHandle) bucketSet(ctx context.Context, entity ACLEntity, role ACLRole) error {
	acl := &raw.BucketAccessControl{
		Bucket: a.bucket,
		Entity: string(entity),
		Role:   string(role),
	}
	_, err := a.c.raw.BucketAccessControls.Update(a.bucket, string(entity), acl).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error updating bucket ACL entry for bucket %q, entity %q: %v", a.bucket, entity, err)
	}
	return nil
}

func (a *ACLHandle) bucketDelete(ctx context.Context, entity ACLEntity) error {
	err := a.c.raw.BucketAccessControls.Delete(a.bucket, string(entity)).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error deleting bucket ACL entry for bucket %q, entity %q: %v", a.bucket, entity, err)
	}
	return nil
}

func (a *ACLHandle) objectList(ctx context.Context) ([]ACLRule, error) {
	acls, err := a.c.raw.ObjectAccessControls.List(a.bucket, a.object).Context(ctx).Do()
	if err != nil {
		return nil, fmt.Errorf("storage: error listing object ACL for bucket %q, file %q: %v", a.bucket, a.object, err)
	}
	return toACLRules(acls.Items), nil
}

func (a *ACLHandle) objectSet(ctx context.Context, entity ACLEntity, role ACLRole) error {
	acl := &raw.ObjectAccessControl{
		Bucket: a.bucket,
		Entity: string(entity),
		Role:   string(role),
	}
	_, err := a.c.raw.ObjectAccessControls.Update(a.bucket, a.object, string(entity), acl).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error updating object ACL entry for bucket %q, file %q, entity %q: %v", a.bucket, a.object, entity, err)
	}
	return nil
}

func (a *ACLHandle) objectDelete(ctx context.Context, entity ACLEntity) error {
	err := a.c.raw.ObjectAccessControls.Delete(a.bucket, a.object, string(entity)).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("storage: error deleting object ACL entry for bucket %q, file %q, entity %q: %v", a.bucket, a.object, entity, err)
	}
	return nil
}

func toACLRules(items []interface{}) []ACLRule {
	r := make([]ACLRule, 0, len(items))
	for _, v := range items {
		if m, ok := v.(map[string]interface{}); ok {
			entity, ok1 := m["entity"].(string)
			role, ok2 := m["role"].(string)
			if ok1 && ok2 {
				r = append(r, ACLRule{Entity: ACLEntity(entity), Role: ACLRole(role)})
			}
		}
	}
	return r
}
