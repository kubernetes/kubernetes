/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
)

// ResourceAttributesFrom combines the API object information and the user.Info from the context to build a full authorizer.AttributesRecord for resource access
func ResourceAttributesFrom(user user.Info, in authorizationapi.ResourceAttributes) authorizer.AttributesRecord {
	return authorizer.AttributesRecord{
		User:            user,
		Verb:            in.Verb,
		Namespace:       in.Namespace,
		APIGroup:        in.Group,
		Resource:        in.Resource,
		ResourceRequest: true,
	}
}

// NonResourceAttributesFrom combines the API object information and the user.Info from the context to build a full authorizer.AttributesRecord for non resource access
func NonResourceAttributesFrom(user user.Info, in authorizationapi.NonResourceAttributes) authorizer.AttributesRecord {
	return authorizer.AttributesRecord{
		User:            user,
		ResourceRequest: false,
		Path:            in.Path,
	}
}
