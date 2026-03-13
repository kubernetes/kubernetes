/*
Copyright 2023 The logr Authors.

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

package logr

// contextKey is how we find Loggers in a context.Context. With Go < 1.21,
// the value is always a Logger value. With Go >= 1.21, the value can be a
// Logger value or a slog.Logger pointer.
type contextKey struct{}

// notFoundError exists to carry an IsNotFound method.
type notFoundError struct{}

func (notFoundError) Error() string {
	return "no logr.Logger was present"
}

func (notFoundError) IsNotFound() bool {
	return true
}
