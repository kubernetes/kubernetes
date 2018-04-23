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

package validate

// TODO: define this as package validate/internal
// This must be done while keeping CI intact with all tests and test coverage

import (
	"reflect"
	"strconv"
	"strings"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
)

// Helpers available at the package level
var (
	pathHelp     *pathHelper
	valueHelp    *valueHelper
	errorHelp    *errorHelper
	paramHelp    *paramHelper
	responseHelp *responseHelper
)

type errorHelper struct {
	// A collection of unexported helpers for error construction
}

func (h *errorHelper) sErr(err errors.Error) *Result {
	// Builds a Result from standard errors.Error
	return &Result{Errors: []error{err}}
}

func (h *errorHelper) addPointerError(res *Result, err error, ref string, fromPath string) *Result {
	// Provides more context on error messages
	// reported by the jsoinpointer package by altering the passed Result
	if err != nil {
		res.AddErrors(cannotResolveRefMsg(fromPath, ref, err))
	}
	return res
}

type pathHelper struct {
	// A collection of unexported helpers for path validation
}

func (h *pathHelper) stripParametersInPath(path string) string {
	// Returns a path stripped from all path parameters, with multiple or trailing slashes removed.
	//
	// Stripping is performed on a slash-separated basis, e.g '/a{/b}' remains a{/b} and not /a.
	//  - Trailing "/" make a difference, e.g. /a/ !~ /a (ex: canary/bitbucket.org/swagger.json)
	//  - presence or absence of a parameter makes a difference, e.g. /a/{log} !~ /a/ (ex: canary/kubernetes/swagger.json)

	// Regexp to extract parameters from path, with surrounding {}.
	// NOTE: important non-greedy modifier
	rexParsePathParam := mustCompileRegexp(`{[^{}]+?}`)
	strippedSegments := []string{}

	for _, segment := range strings.Split(path, "/") {
		strippedSegments = append(strippedSegments, rexParsePathParam.ReplaceAllString(segment, "X"))
	}
	return strings.Join(strippedSegments, "/")
}

func (h *pathHelper) extractPathParams(path string) (params []string) {
	// Extracts all params from a path, with surrounding "{}"
	rexParsePathParam := mustCompileRegexp(`{[^{}]+?}`)

	for _, segment := range strings.Split(path, "/") {
		for _, v := range rexParsePathParam.FindAllStringSubmatch(segment, -1) {
			params = append(params, v...)
		}
	}
	return
}

type valueHelper struct {
	// A collection of unexported helpers for value validation
}

func (h *valueHelper) asInt64(val interface{}) int64 {
	// Number conversion function for int64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int64(v.Uint())
	case reflect.Float32, reflect.Float64:
		return int64(v.Float())
	default:
		//panic("Non numeric value in asInt64()")
		return 0
	}
}

func (h *valueHelper) asUint64(val interface{}) uint64 {
	// Number conversion function for uint64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return uint64(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint()
	case reflect.Float32, reflect.Float64:
		return uint64(v.Float())
	default:
		//panic("Non numeric value in asUint64()")
		return 0
	}
}

// Same for unsigned floats
func (h *valueHelper) asFloat64(val interface{}) float64 {
	// Number conversion function for float64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint())
	case reflect.Float32, reflect.Float64:
		return v.Float()
	default:
		//panic("Non numeric value in asFloat64()")
		return 0
	}
}

type paramHelper struct {
	// A collection of unexported helpers for parameters resolution
}

func (h *paramHelper) safeExpandedParamsFor(path, method, operationID string, res *Result, s *SpecValidator) (params []spec.Parameter) {
	operation, ok := s.analyzer.OperationFor(method, path)
	if ok {
		// expand parameters first if necessary
		resolvedParams := []spec.Parameter{}
		for _, ppr := range operation.Parameters {
			resolvedParam, red := h.resolveParam(path, method, operationID, &ppr, s)
			res.Merge(red)
			if resolvedParam != nil {
				resolvedParams = append(resolvedParams, *resolvedParam)
			}
		}
		// remove params with invalid expansion from Slice
		operation.Parameters = resolvedParams

		for _, ppr := range s.analyzer.SafeParamsFor(method, path,
			func(p spec.Parameter, err error) bool {
				// since params have already been expanded, there are few causes for error
				res.AddErrors(someParametersBrokenMsg(path, method, operationID))
				// original error from analyzer
				res.AddErrors(err)
				return true
			}) {
			params = append(params, ppr)
		}
	}
	return
}

func (h *paramHelper) resolveParam(path, method, operationID string, param *spec.Parameter, s *SpecValidator) (*spec.Parameter, *Result) {
	// Expand parameter with $ref if needed
	res := new(Result)

	if param.Ref.String() != "" {
		err := spec.ExpandParameter(param, s.spec.SpecFilePath())
		if err != nil { // Safeguard
			// NOTE: we may enter enter here when the whole parameter is an unresolved $ref
			refPath := strings.Join([]string{"\"" + path + "\"", method}, ".")
			errorHelp.addPointerError(res, err, param.Ref.String(), refPath)
			return nil, res
		}
		res.Merge(h.checkExpandedParam(param, param.Name, param.In, operationID))
	}
	return param, res
}

func (h *paramHelper) checkExpandedParam(pr *spec.Parameter, path, in, operation string) *Result {
	// Secure parameter structure after $ref resolution
	res := new(Result)
	simpleZero := spec.SimpleSchema{}
	// Try to explain why... best guess
	if pr.In == "body" && pr.SimpleSchema != simpleZero {
		// Most likely, a $ref with a sibling is an unwanted situation: in itself this is a warning...
		// but we detect it because of the following error:
		// schema took over Parameter for an unexplained reason
		res.AddWarnings(refShouldNotHaveSiblingsMsg(path, operation))
		res.AddErrors(invalidParameterDefinitionMsg(path, in, operation))
	} else if pr.In != "body" && pr.Schema != nil {
		res.AddWarnings(refShouldNotHaveSiblingsMsg(path, operation))
		res.AddErrors(invalidParameterDefinitionAsSchemaMsg(path, in, operation))
	} else if (pr.In == "body" && pr.Schema == nil) || (pr.In != "body" && pr.SimpleSchema == simpleZero) { // Safeguard
		// Other unexpected mishaps
		res.AddErrors(invalidParameterDefinitionMsg(path, in, operation))
	}
	return res
}

type responseHelper struct {
	// A collection of unexported helpers for response resolution
}

func (r *responseHelper) expandResponseRef(response *spec.Response, path string, s *SpecValidator) (*spec.Response, *Result) {
	// Expand response with $ref if needed
	res := new(Result)
	if response.Ref.String() != "" {
		err := spec.ExpandResponse(response, s.spec.SpecFilePath())
		if err != nil { // Safeguard
			// NOTE: we may enter here when the whole response is an unresolved $ref.
			errorHelp.addPointerError(res, err, response.Ref.String(), path)
			return nil, res
		}
	}
	return response, res
}

func (r *responseHelper) responseMsgVariants(responseType string, responseCode int) (responseName, responseCodeAsStr string) {
	// Path variants for messages
	if responseType == "default" {
		responseCodeAsStr = "default"
		responseName = "default response"
	} else {
		responseCodeAsStr = strconv.Itoa(responseCode)
		responseName = "response " + responseCodeAsStr
	}
	return
}
