/*
Copyright 2020 The Kubernetes Authors.

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

package fieldmanager

import "context"

type key int

const fieldmanagerKey key = iota

// ToContext adds a field manager to the given context
func ToContext(ctx context.Context, fm *FieldManager) context.Context {
	return context.WithValue(ctx, fieldmanagerKey, fm)
}

// FromContext retrieves the field manager from the given context, if the
// conversion to *FieldManager fails it returns nil
func FromContext(ctx context.Context) *FieldManager {
	fmRaw := ctx.Value(fieldmanagerKey)
	fm, ok := fmRaw.(*FieldManager)
	if !ok {
		return nil
	}
	return fm
}
