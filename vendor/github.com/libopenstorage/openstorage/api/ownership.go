/*
Package ownership manages access to resources
Copyright 2019 Portworx

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package api

import (
	"context"

	"github.com/libopenstorage/openstorage/pkg/auth"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// AdminGroup is the value that can be set in the token claims Group which
	// gives the user access to any resource
	AdminGroup = "*"
)

// OwnershipSetUsernameFromContext is used to create a new ownership object for
// a volume. It takes an ownership value if passed in by the user, then
// sets the `owner` value to the user name referred to in the user context
func OwnershipSetUsernameFromContext(ctx context.Context, srcOwnership *Ownership) *Ownership {
	// Check if the context has information about the user. If not,
	// then security is not enabled.
	if userinfo, ok := auth.NewUserInfoFromContext(ctx); ok {
		// Guest user cannot provide ownership
		if userinfo.IsGuest() {
			return nil
		}

		// Merge the previous acls which may have been set by the user
		var acls *Ownership_AccessControl
		if srcOwnership != nil {
			acls = srcOwnership.GetAcls()
		}

		return &Ownership{
			Owner: userinfo.Username,
			Acls:  acls,
		}
	}

	return srcOwnership
}

// IsPermittedByContext returns true if the user captured in
// the context has permission to access the resource
func (o *Ownership) IsPermittedByContext(
	ctx context.Context,
	accessType Ownership_AccessType) bool {

	// If no ownership is there then it is public
	if o == nil {
		return true
	}

	// Volume is not public, check permission
	if userinfo, ok := auth.NewUserInfoFromContext(ctx); ok {
		// Check Access
		return o.IsPermitted(userinfo, accessType)
	} else {
		// There is no user information in the context so
		// authorization is not running
		return true
	}
}

// IsPermitted returns true if the user has access to the resource
// according to the ownership. If there is no owner, then it is public
func (o *Ownership) IsPermitted(
	user *auth.UserInfo,
	accessType Ownership_AccessType,
) bool {
	// There is no owner, so it is a public resource
	if o.IsPublic(accessType) {
		return true
	}

	// If we are missing user information then do not allow.
	// It is ok for the the user claims to have an empty Groups setting
	if user == nil ||
		len(user.Username) == 0 {
		return false
	}

	if o.IsOwner(user) ||
		o.IsUserAllowedByGroup(user, accessType) ||
		o.IsUserAllowedByCollaborators(user, accessType) {
		return true
	}

	return false
}

// GetGroups returns the groups in the ownership
func (o *Ownership) GetGroups() map[string]Ownership_AccessType {
	if o.GetAcls() == nil {
		return nil
	}
	return o.GetAcls().GetGroups()
}

// GetCollaborators returns the collaborators in the ownership
func (o *Ownership) GetCollaborators() map[string]Ownership_AccessType {
	if o.GetAcls() == nil {
		return nil
	}
	return o.GetAcls().GetCollaborators()
}

// IsUserAllowedByGroup returns true if the user is allowed access
// by belonging to the appropriate group
func (o *Ownership) IsUserAllowedByGroup(
	user *auth.UserInfo,
	accessType Ownership_AccessType,
) bool {

	// Allow if it is the admin user for any group
	if o.IsAdminByUser(user) {
		return true
	}

	ownergroups := o.GetGroups()
	if len(ownergroups) == 0 {
		return false
	}

	// Check each of the groups from the user
	for _, group := range user.Claims.Groups {
		// Check if the user group has permission
		if a, ok := ownergroups[group]; ok {
			return a.isAccessPermitted(accessType)
		}
	}

	// Check if any group is allowed
	if a, ok := ownergroups["*"]; ok {
		return a.isAccessPermitted(accessType)
	}

	return false
}

// IsUserAllowedByCollaborators returns true if the user is allowed access
// because they are part of the collaborators list
func (o *Ownership) IsUserAllowedByCollaborators(
	user *auth.UserInfo,
	accessType Ownership_AccessType,
) bool {
	collaborators := o.GetCollaborators()
	if len(collaborators) == 0 {
		return false
	}

	// Check each of the groups from the user
	if a, ok := collaborators[user.Username]; ok {
		return a.isAccessPermitted(accessType)
	}

	// Check any user is allowed
	if a, ok := collaborators["*"]; ok {
		return a.isAccessPermitted(accessType)
	}

	return false
}

// HasAnOwner returns true if the resource has an owner
func (o *Ownership) HasAnOwner() bool {
	return len(o.Owner) != 0
}

// IsAccessPermittedByPublic returns true if access is permitted for public user
func (o *Ownership) IsAccessPermittedByPublic(accessType Ownership_AccessType) bool {
	return (o.Acls != nil && o.Acls.Public != nil &&
		o.Acls.Public.Type.isAccessPermitted(accessType))
}

// IsPublic returns true if public access is set or
// there is no ownership in this resource
func (o *Ownership) IsPublic(accessType Ownership_AccessType) bool {
	return o.IsAccessPermittedByPublic(accessType) || !o.HasAnOwner()
}

// IsOwner returns if the user is the owner of the resource
func (o *Ownership) IsOwner(user *auth.UserInfo) bool {
	return o.Owner == user.Username
}

// IsAdminByUser returns true if the user is an ownership admin, meaning,
// that they belong to any group
func (o *Ownership) IsAdminByUser(user *auth.UserInfo) bool {
	return IsAdminByUser(user)
}

// Update can be used to update an ownership with new ownership information. It
// takes into account who is trying to change the ownership values
func (o *Ownership) Update(newownerInfo *Ownership, user *auth.UserInfo) error {
	if user == nil {
		// There is no auth, just copy the whole thing
		*o = *newownerInfo
	} else {
		// Auth is enabled

		// Only the owner, user with access type admin,
		// or admin can change the group
		if user.Username != o.Owner &&
			!o.IsAdminByUser(user) &&
			!o.IsPermitted(user, Ownership_Admin) {
			return status.Error(codes.PermissionDenied,
				"Only owner or those with admin access type can update volume acls")
		}

		// Only the admin can change the owner
		if newownerInfo.HasAnOwner() {
			if o.IsAdminByUser(user) {
				o.Owner = newownerInfo.Owner
			} else {
				return status.Error(codes.PermissionDenied,
					"Only the administrator can change the owner of the resource")
			}
		}
		o.Acls = newownerInfo.GetAcls()
	}
	return nil
}

// IsMatch returns true if the ownership has at least one similar
// owner, group, or collaborator
func (o *Ownership) IsMatch(check *Ownership) bool {
	if check == nil {
		return false
	}

	// Check user
	if o.Owner == check.GetOwner() {
		return true
	}
	if check.GetAcls() == nil || o.GetAcls() == nil {
		return false
	}

	// Check groups
	for group := range check.GetAcls().GetGroups() {
		if _, ok := o.GetAcls().GetGroups()[group]; ok {
			return true
		}
	}

	// Check collaborators
	for collaborator := range check.GetAcls().GetCollaborators() {
		if _, ok := o.GetAcls().GetCollaborators()[collaborator]; ok {
			return true
		}
	}

	return false
}

func (a Ownership_AccessType) isAccessPermitted(accessType Ownership_AccessType) bool {
	return a >= accessType
}

func listContains(list []string, s string) bool {
	for _, value := range list {
		if value == s {
			return true
		}
	}
	return false
}

// IsAdminByUser returns true if the user is an ownership admin, meaning,
// that they belong to any group
func IsAdminByUser(user *auth.UserInfo) bool {
	// If there is a user, then auth is enabled
	if user != nil {
		return !user.IsGuest() && listContains(user.Claims.Groups, AdminGroup)
	}

	// No auth enabled, so everyone is an admin
	return true
}

// IsAdminByContext checks if the context userInfo contains admin privileges
func IsAdminByContext(ctx context.Context) bool {
	// Check if the context has information about the user. If not,
	// then security is not enabled.
	if userinfo, ok := auth.NewUserInfoFromContext(ctx); ok {
		return IsAdminByUser(userinfo)
	}
	return true
}
