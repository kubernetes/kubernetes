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

/*Package middleware provides the library with helper functions for serving swagger APIs.

Pseudo middleware handler

  import (
  	"net/http"

  	"github.com/go-openapi/errors"
  	"github.com/gorilla/context"
  )

  func newCompleteMiddleware(ctx *Context) http.Handler {
  	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
  		defer context.Clear(r)

  		// use context to lookup routes
  		if matched, ok := ctx.RouteInfo(r); ok {

  			if len(matched.Authenticators) > 0 {
  				if _, err := ctx.Authorize(r, matched); err != nil {
  					ctx.Respond(rw, r, matched.Produces, matched, err)
  					return
  				}
  			}

  			bound, validation := ctx.BindAndValidate(r, matched)
  			if validation != nil {
  				ctx.Respond(rw, r, matched.Produces, matched, validation)
  				return
  			}

  			result, err := matched.Handler.Handle(bound)
  			if err != nil {
  				ctx.Respond(rw, r, matched.Produces, matched, err)
  				return
  			}

  			ctx.Respond(rw, r, matched.Produces, matched, result)
  			return
  		}

  		// Not found, check if it exists in the other methods first
  		if others := ctx.AllowedMethods(r); len(others) > 0 {
  			ctx.Respond(rw, r, ctx.spec.RequiredProduces(), nil, errors.MethodNotAllowed(r.Method, others))
  			return
  		}
  		ctx.Respond(rw, r, ctx.spec.RequiredProduces(), nil, errors.NotFound("path %s was not found", r.URL.Path))
  	})
  }
*/
package middleware
