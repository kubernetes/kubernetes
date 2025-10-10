/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package xdsclient

import (
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
)

// WatchResource uses xDS to discover the resource associated with the provided
// resource name. The resource type implementation determines how xDS responses
// are are deserialized and validated, as received from the xDS management
// server. Upon receipt of a response from the management server, an
// appropriate callback on the watcher is invoked.
func (c *clientImpl) WatchResource(rType xdsresource.Type, resourceName string, watcher xdsresource.ResourceWatcher) (cancel func()) {
	return c.XDSClient.WatchResource(rType.TypeURL(), resourceName, xdsresource.GenericResourceWatcher(watcher))
}
