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

package authenticatorfactory

import (
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
)

// RequestHeaderConfig wires request-header authentication's dynamic configuration and
// client CA provider. The immutable configuration snapshot is headerrequest.RequestHeaderConfig.
type RequestHeaderConfig struct {
	// Config provides atomic point-in-time snapshots of the request header configuration,
	// i.e. the headers to inspect and the allowed front-proxy client common names. It
	// returns nil when no configuration is available, e.g. because the source configmap
	// has never been read or has been deleted. Consumers must fail closed on a nil snapshot.
	Config headerrequest.RequestHeaderConfigProvider
	// CAContentProvider are the options for verifying incoming connections using mTLS.  Generally this points to CA bundle file which is used verify the identity of the front proxy.
	//	It may produce different options at will.
	CAContentProvider dynamiccertificates.CAContentProvider
}
