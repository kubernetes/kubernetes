/*
Package auth can be used for authentication and authorization
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
package auth

import (
	"context"
)

// Keys to store data in gRPC context. Use these keys to retrieve
// the data from the gRPC context
type InterceptorContextkey string

const (
	// Key to store in the token claims in gRPC context
	InterceptorContextTokenKey InterceptorContextkey = "tokenclaims"
)

// UserInfo contains information about the user taken from the token
type UserInfo struct {
	// Username is the unique id of the user. According to the configuration of
	// the storage system, this could be the 'sub', 'name', or 'email' from
	// the claims in the token.
	Username string
	// Claims holds the claims required by the storage system
	Claims Claims
	// Guest marks whether the user is unauthenticated
	Guest bool
}

// ContextSaveUserInfo saves user information in the context for other functions to consume
func ContextSaveUserInfo(ctx context.Context, u *UserInfo) context.Context {
	return context.WithValue(ctx, InterceptorContextTokenKey, u)
}

// NewUserInfoFromContext returns user information in the context if available.
// If not available means that the system is running without auth.
func NewUserInfoFromContext(ctx context.Context) (*UserInfo, bool) {
	u, ok := ctx.Value(InterceptorContextTokenKey).(*UserInfo)
	return u, ok
}

// NewGuestUser creates UserInfo for the system guest user
func NewGuestUser() *UserInfo {
	return &UserInfo{
		Claims: Claims{
			Roles: []string{systemGuestRoleName},
		},
		Guest: true,
	}
}

// IsGuest returns whether or not the UserInfo is for a guest user
func (ui *UserInfo) IsGuest() bool {
	return ui.Guest
}
