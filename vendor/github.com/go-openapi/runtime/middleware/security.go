// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package middleware

import "net/http"

func newSecureAPI(ctx *Context, next http.Handler) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		route, _ := ctx.RouteInfo(r)
		if len(route.Authenticators) == 0 {
			next.ServeHTTP(rw, r)
			return
		}

		if _, err := ctx.Authorize(r, route); err != nil {
			ctx.Respond(rw, r, route.Produces, route, err)
			return
		}

		next.ServeHTTP(rw, r)
	})
}
