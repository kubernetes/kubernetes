/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount

import (
	"k8s.io/api/core/v1"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
)

// UserInfo returns a user.Info interface for the given namespace, service account name and UID
func UserInfo(namespace, name, uid string) user.Info {
	return &user.DefaultInfo{
		Name:   apiserverserviceaccount.MakeUsername(namespace, name),
		UID:    uid,
		Groups: apiserverserviceaccount.MakeGroupNames(namespace),
	}
}

// IsServiceAccountToken returns true if the secret is a valid api token for the service account
func IsServiceAccountToken(secret *v1.Secret, sa *v1.ServiceAccount) bool {
	if secret.Type != v1.SecretTypeServiceAccountToken {
		return false
	}

	name := secret.Annotations[v1.ServiceAccountNameKey]
	uid := secret.Annotations[v1.ServiceAccountUIDKey]
	if name != sa.Name {
		// Name must match
		return false
	}
	if len(uid) > 0 && uid != string(sa.UID) {
		// If UID is specified, it must match
		return false
	}

	return true
}

// TODO: remove the duplicate code
// InternalIsServiceAccountToken returns true if the secret is a valid api token for the service account
func InternalIsServiceAccountToken(secret *api.Secret, sa *api.ServiceAccount) bool {
	if secret.Type != api.SecretTypeServiceAccountToken {
		return false
	}

	name := secret.Annotations[api.ServiceAccountNameKey]
	uid := secret.Annotations[api.ServiceAccountUIDKey]
	if name != sa.Name {
		// Name must match
		return false
	}
	if len(uid) > 0 && uid != string(sa.UID) {
		// If UID is specified, it must match
		return false
	}

	return true
}
