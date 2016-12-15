// Copyright 2016 CoreOS, Inc.
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

package fileutil

type Lock interface {
	// Name returns the name of the file.
	Name() string
	// TryLock acquires exclusivity on the lock without blocking.
	TryLock() error
	// Lock acquires exclusivity on the lock.
	Lock() error
	// Unlock unlocks the lock.
	Unlock() error
	// Destroy should be called after Unlock to clean up
	// the resources.
	Destroy() error
}
