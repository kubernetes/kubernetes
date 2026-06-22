// Copyright 2026 The etcd Authors
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

package authpb

const (
	// READ is an alias of Permission_READ
	// Deprecated: use Permission_READ instead. Will be removed in v3.8.
	READ = Permission_READ
	// WRITE is an alias of Permission_WRITE
	// Deprecated: use Permission_WRITE instead. Will be removed in v3.8.
	WRITE = Permission_WRITE
	// READWRITE is an alias of Permission_READWRITE
	// Deprecated: use Permission_READWRITE instead. Will be removed in v3.8.
	READWRITE = Permission_READWRITE
)
