/*
Copyright 2015 The Kubernetes Authors.

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

package types

// NodeName is a type that holds a api.Node's Name identifier.
// Being a type captures intent and helps make sure that the node name
// is not confused with similar concepts (the hostname, the cloud provider id,
// the cloud provider name etc)
//
// To clarify the various types:
//
//   - Node.Name is the Name field of the Node in the API.  This should be stored in a NodeName.
//     Unfortunately, because Name is part of ObjectMeta, we can't store it as a NodeName at the API level.
//
//   - Hostname is the hostname of the local machine (from uname -n).
//     However, some components allow the user to pass in a --hostname-override flag,
//     which will override this in most places. In the absence of anything more meaningful,
//     kubelet will use Hostname as the Node.Name when it creates the Node.
//
// * The cloudproviders have the own names: GCE has InstanceName, AWS has InstanceId.
//
//	For GCE, InstanceName is the Name of an Instance object in the GCE API.  On GCE, Instance.Name becomes the
//	Hostname, and thus it makes sense also to use it as the Node.Name.  But that is GCE specific, and it is up
//	to the cloudprovider how to do this mapping.
//
//	For AWS, the InstanceID is not yet suitable for use as a Node.Name, so we actually use the
//	PrivateDnsName for the Node.Name.  And this is _not_ always the same as the hostname: if
//	we are using a custom DHCP domain it won't be.
type NodeName string
