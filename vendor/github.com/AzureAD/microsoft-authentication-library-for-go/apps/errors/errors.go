// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package errors

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"

	"github.com/kylelemons/godebug/pretty"
)

var prettyConf = &pretty.Config{
	IncludeUnexported: false,
	SkipZeroFields:    true,
	TrackCycles:       true,
	Formatter: map[reflect.Type]interface{}{
		reflect.TypeOf((*io.Reader)(nil)).Elem(): func(r io.Reader) string {
			b, err := io.ReadAll(r)
			if err != nil {
				return "could not read io.Reader content"
			}
			return string(b)
		},
	},
}

type verboser interface {
	Verbose() string
}

// Verbose prints the most verbose error that the error message has.
func Verbose(err error) string {
	build := strings.Builder{}
	for {
		if err == nil {
			break
		}
		if v, ok := err.(verboser); ok {
			build.WriteString(v.Verbose())
		} else {
			build.WriteString(err.Error())
		}
		err = errors.Unwrap(err)
	}
	return build.String()
}

// New is equivalent to errors.New().
func New(text string) error {
	return errors.New(text)
}

// CallErr represents an HTTP call error. Has a Verbose() method that allows getting the
// http.Request and Response objects. Implements error.
type CallErr struct {
	Req *http.Request
	// Resp contains response body
	Resp *http.Response
	Err  error
}

// Errors implements error.Error().
func (e CallErr) Error() string {
	return e.Err.Error()
}

// Verbose prints a versbose error message with the request or response.
func (e CallErr) Verbose() string {
	e.Resp.Request = nil // This brings in a bunch of TLS crap we don't need
	e.Resp.TLS = nil     // Same
	return fmt.Sprintf("%s:\nRequest:\n%s\nResponse:\n%s", e.Err, prettyConf.Sprint(e.Req), prettyConf.Sprint(e.Resp))
}

// Is reports whether any error in errors chain matches target.
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As finds the first error in errors chain that matches target,
// and if so, sets target to that error value and returns true.
// Otherwise, it returns false.
func As(err error, target interface{}) bool {
	return errors.As(err, target)
}
