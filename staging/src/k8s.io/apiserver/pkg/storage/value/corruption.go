/*
Copyright The Kubernetes Authors.

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

package value

import "errors"

// IsDataCorruption reports whether an error from TransformFromStorage means the
// stored bytes are damaged, so deleting the object unsafely is an acceptable
// recovery. If err, or an error it wraps, has a DataCorruption method, that method
// decides. An error without one counts as corruption, since KEP-3926 treats every
// failed transform that way.
func IsDataCorruption(err error) bool {
	var c interface{ DataCorruption() bool }
	if errors.As(err, &c) {
		return c.DataCorruption()
	}
	return err != nil
}
