// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/*
   Copyright The containerd Authors.

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

package namespaces

import (
	"context"

	"github.com/containerd/ttrpc"
)

const (
	// TTRPCHeader defines the header name for specifying a containerd namespace
	TTRPCHeader = "containerd-namespace-ttrpc"
)

func copyMetadata(src ttrpc.MD) ttrpc.MD {
	md := ttrpc.MD{}
	for k, v := range src {
		md[k] = append(md[k], v...)
	}
	return md
}

func withTTRPCNamespaceHeader(ctx context.Context, namespace string) context.Context {
	md, ok := ttrpc.GetMetadata(ctx)
	if !ok {
		md = ttrpc.MD{}
	} else {
		md = copyMetadata(md)
	}
	md.Set(TTRPCHeader, namespace)
	return ttrpc.WithMetadata(ctx, md)
}

func fromTTRPCHeader(ctx context.Context) (string, bool) {
	return ttrpc.GetMetadataValue(ctx, TTRPCHeader)
}
