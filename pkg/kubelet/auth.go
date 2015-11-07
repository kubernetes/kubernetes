/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authorizer"
)

// KubeletAuth implements AuthInterface
type KubeletAuth struct {
	// authenticator identifies the user for requests to the Kubelet API
	authenticator.Request
	// authorizerAttributeGetter builds authorization.Attributes for a request to the Kubelet API
	authorizer.RequestAttributesGetter
	// authorizer determines whether a given authorization.Attributes is allowed
	authorizer.Authorizer
}

// NewKubeletAuth returns a kubelet.AuthInterface composed of the given authenticator, attribute getter, and authorizer
func NewKubeletAuth(authenticator authenticator.Request, authorizerAttributeGetter authorizer.RequestAttributesGetter, authorizer authorizer.Authorizer) AuthInterface {
	return &KubeletAuth{authenticator, authorizerAttributeGetter, authorizer}
}
