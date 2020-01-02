/*
Copyright 2018 The Kubernetes Authors.

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

package authenticator

import "context"

// Audiences is a container for the Audiences of a token.
type Audiences []string

// The key type is unexported to prevent collisions
type key int

const (
	// audiencesKey is the context key for request audiences.
	audiencesKey key = iota
)

// WithAudiences returns a context that stores a request's expected audiences.
func WithAudiences(ctx context.Context, auds Audiences) context.Context {
	return context.WithValue(ctx, audiencesKey, auds)
}

// AudiencesFrom returns a request's expected audiences stored in the request context.
func AudiencesFrom(ctx context.Context) (Audiences, bool) {
	auds, ok := ctx.Value(audiencesKey).(Audiences)
	return auds, ok
}

// Has checks if Audiences contains a specific audiences.
func (a Audiences) Has(taud string) bool {
	for _, aud := range a {
		if aud == taud {
			return true
		}
	}
	return false
}

// Intersect intersects Audiences with a target Audiences and returns all
// elements in both.
func (a Audiences) Intersect(tauds Audiences) Audiences {
	selected := Audiences{}
	for _, taud := range tauds {
		if a.Has(taud) {
			selected = append(selected, taud)
		}
	}
	return selected
}
