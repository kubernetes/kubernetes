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

import (
	"mime"
	"net/http"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/swag"
)

// NewValidation starts a new validation middleware
func newValidation(ctx *Context, next http.Handler) http.Handler {

	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		matched, _ := ctx.RouteInfo(r)
		_, result := ctx.BindAndValidate(r, matched)

		if result != nil {
			ctx.Respond(rw, r, matched.Produces, matched, result)
			return
		}

		next.ServeHTTP(rw, r)
	})
}

type validation struct {
	context *Context
	result  []error
	request *http.Request
	route   *MatchedRoute
	bound   map[string]interface{}
}

type untypedBinder map[string]interface{}

func (ub untypedBinder) BindRequest(r *http.Request, route *MatchedRoute, consumer runtime.Consumer) error {
	if err := route.Binder.Bind(r, route.Params, consumer, ub); err != nil {
		return err
	}
	return nil
}

// ContentType validates the content type of a request
func validateContentType(allowed []string, actual string) error {
	if len(allowed) == 0 {
		return nil
	}
	mt, _, err := mime.ParseMediaType(actual)
	if err != nil {
		return errors.InvalidContentType(actual, allowed)
	}
	if swag.ContainsStringsCI(allowed, mt) {
		return nil
	}
	return errors.InvalidContentType(actual, allowed)
}

func validateRequest(ctx *Context, request *http.Request, route *MatchedRoute) *validation {
	validate := &validation{
		context: ctx,
		request: request,
		route:   route,
		bound:   make(map[string]interface{}),
	}

	validate.contentType()
	if len(validate.result) == 0 {
		validate.responseFormat()
	}
	if len(validate.result) == 0 {
		validate.parameters()
	}

	return validate
}

func (v *validation) parameters() {
	if result := v.route.Binder.Bind(v.request, v.route.Params, v.route.Consumer, v.bound); result != nil {
		if result.Error() == "validation failure list" {
			for _, e := range result.(*errors.Validation).Value.([]interface{}) {
				v.result = append(v.result, e.(error))
			}
			return
		}
		v.result = append(v.result, result)
	}
}

func (v *validation) contentType() {
	if len(v.result) == 0 && runtime.HasBody(v.request) {
		ct, _, err := v.context.ContentType(v.request)
		if err != nil {
			v.result = append(v.result, err)
		}
		if len(v.result) == 0 {
			if err := validateContentType(v.route.Consumes, ct); err != nil {
				v.result = append(v.result, err)
			}
		}
		if ct != "" && v.route.Consumer == nil {
			cons, ok := v.route.Consumers[ct]
			if !ok {
				v.result = append(v.result, errors.New(500, "no consumer registered for %s", ct))
			} else {
				v.route.Consumer = cons
			}
		}
	}
}

func (v *validation) responseFormat() {
	if str := v.context.ResponseFormat(v.request, v.route.Produces); str == "" && runtime.HasBody(v.request) {
		v.result = append(v.result, errors.InvalidResponseFormat(v.request.Header.Get(runtime.HeaderAccept), v.route.Produces))
	}
}
