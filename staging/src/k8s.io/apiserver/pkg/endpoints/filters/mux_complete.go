/*
Copyright 2021 The Kubernetes Authors.

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

package filters

import (
	"context"
	"net/http"
)

type muxCompleteProtectionKeyType int

const (
	// muxCompleteProtectionKey is a key under which a protection signal for all requests made before the server have installed all known HTTP paths is stored in the request's context
	muxCompleteProtectionKey muxCompleteProtectionKeyType = iota
)

// HasMuxCompleteProtectionKey checks if the context contains muxCompleteProtectionKey.
// The presence of the key indicates the request has been made when the HTTP paths weren't installed.
func HasMuxCompleteProtectionKey(ctx context.Context) bool {
	muxCompleteProtectionKeyValue, _ := ctx.Value(muxCompleteProtectionKey).(string)
	return len(muxCompleteProtectionKeyValue) != 0
}

// WithMuxCompleteProtection puts the muxCompleteProtectionKey in the context if a request has been made before muxCompleteSignal has been ready.
// Putting the key protect us from returning a 404 response instead of a 503.
// It is especially important for controllers like GC and NS since they act on 404s.
//
// The presence of the key is checked in the NotFoundHandler (staging/src/k8s.io/apiserver/pkg/util/notfoundhandler/not_found_handler.go)
//
// The race may happen when a request reaches the NotFoundHandler because not all paths have been registered in the mux
// but when the registered checks are examined in the handler they indicate that the paths have been actually installed.
// In that case, the presence of the key will make the handler return 503 instead of 404.
func WithMuxCompleteProtection(handler http.Handler, muxCompleteSignal <-chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if muxCompleteSignal != nil && !isClosed(muxCompleteSignal) {
			req = req.WithContext(context.WithValue(req.Context(), muxCompleteProtectionKey, "MuxInstallationNotComplete"))
		}
		handler.ServeHTTP(w, req)
	})
}

// isClosed is a convenience function that simply check if the given chan has been closed
func isClosed(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}
