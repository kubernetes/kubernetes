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

package adapters

// These adapters provide a way to get the fields of the role regardless of the API version of the object.
// This is needed so that the RBAC authorizer can cleanly support multiple API versions with a single implementation
// but only depend upon the external clients built into client-go.  That is needed so that the generic API server
// which provides this as an authorization option can be broken out into a separate repo that doesn't depend
// upon the main kube repo.  This really is the *exact* case for which internal versions were created.  I have logic
// that I don't want to duplicate, that is fully convertible between versions.  I just want a version that I support
// negotiated with the server and then convert to the internal representation so that I can use the values.
