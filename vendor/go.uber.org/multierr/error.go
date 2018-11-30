// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Package multierr allows combining one or more errors together.
//
// Overview
//
// Errors can be combined with the use of the Combine function.
//
// 	multierr.Combine(
// 		reader.Close(),
// 		writer.Close(),
// 		conn.Close(),
// 	)
//
// If only two errors are being combined, the Append function may be used
// instead.
//
// 	err = multierr.Append(reader.Close(), writer.Close())
//
// This makes it possible to record resource cleanup failures from deferred
// blocks with the help of named return values.
//
// 	func sendRequest(req Request) (err error) {
// 		conn, err := openConnection()
// 		if err != nil {
// 			return err
// 		}
// 		defer func() {
// 			err = multierr.Append(err, conn.Close())
// 		}()
// 		// ...
// 	}
//
// The underlying list of errors for a returned error object may be retrieved
// with the Errors function.
//
// 	errors := multierr.Errors(err)
// 	if len(errors) > 0 {
// 		fmt.Println("The following errors occurred:")
// 	}
//
// Advanced Usage
//
// Errors returned by Combine and Append MAY implement the following
// interface.
//
// 	type errorGroup interface {
// 		// Returns a slice containing the underlying list of errors.
// 		//
// 		// This slice MUST NOT be modified by the caller.
// 		Errors() []error
// 	}
//
// Note that if you need access to list of errors behind a multierr error, you
// should prefer using the Errors function. That said, if you need cheap
// read-only access to the underlying errors slice, you can attempt to cast
// the error to this interface. You MUST handle the failure case gracefully
// because errors returned by Combine and Append are not guaranteed to
// implement this interface.
//
// 	var errors []error
// 	group, ok := err.(errorGroup)
// 	if ok {
// 		errors = group.Errors()
// 	} else {
// 		errors = []error{err}
// 	}
package multierr

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"sync"

	"go.uber.org/atomic"
)

var (
	// Separator for single-line error messages.
	_singlelineSeparator = []byte("; ")

	_newline = []byte("\n")

	// Prefix for multi-line messages
	_multilinePrefix = []byte("the following errors occurred:")

	// Prefix for the first and following lines of an item in a list of
	// multi-line error messages.
	//
	// For example, if a single item is:
	//
	// 	foo
	// 	bar
	//
	// It will become,
	//
	// 	 -  foo
	// 	    bar
	_multilineSeparator = []byte("\n -  ")
	_multilineIndent    = []byte("    ")
)

// _bufferPool is a pool of bytes.Buffers.
var _bufferPool = sync.Pool{
	New: func() interface{} {
		return &bytes.Buffer{}
	},
}

type errorGroup interface {
	Errors() []error
}

// Errors returns a slice containing zero or more errors that the supplied
// error is composed of. If the error is nil, the returned slice is empty.
//
// 	err := multierr.Append(r.Close(), w.Close())
// 	errors := multierr.Errors(err)
//
// If the error is not composed of other errors, the returned slice contains
// just the error that was passed in.
//
// Callers of this function are free to modify the returned slice.
func Errors(err error) []error {
	if err == nil {
		return nil
	}

	// Note that we're casting to multiError, not errorGroup. Our contract is
	// that returned errors MAY implement errorGroup. Errors, however, only
	// has special behavior for multierr-specific error objects.
	//
	// This behavior can be expanded in the future but I think it's prudent to
	// start with as little as possible in terms of contract and possibility
	// of misuse.
	eg, ok := err.(*multiError)
	if !ok {
		return []error{err}
	}

	errors := eg.Errors()
	result := make([]error, len(errors))
	copy(result, errors)
	return result
}

// multiError is an error that holds one or more errors.
//
// An instance of this is guaranteed to be non-empty and flattened. That is,
// none of the errors inside multiError are other multiErrors.
//
// multiError formats to a semi-colon delimited list of error messages with
// %v and with a more readable multi-line format with %+v.
type multiError struct {
	copyNeeded atomic.Bool
	errors     []error
}

var _ errorGroup = (*multiError)(nil)

// Errors returns the list of underlying errors.
//
// This slice MUST NOT be modified.
func (merr *multiError) Errors() []error {
	if merr == nil {
		return nil
	}
	return merr.errors
}

func (merr *multiError) Error() string {
	if merr == nil {
		return ""
	}

	buff := _bufferPool.Get().(*bytes.Buffer)
	buff.Reset()

	merr.writeSingleline(buff)

	result := buff.String()
	_bufferPool.Put(buff)
	return result
}

func (merr *multiError) Format(f fmt.State, c rune) {
	if c == 'v' && f.Flag('+') {
		merr.writeMultiline(f)
	} else {
		merr.writeSingleline(f)
	}
}

func (merr *multiError) writeSingleline(w io.Writer) {
	first := true
	for _, item := range merr.errors {
		if first {
			first = false
		} else {
			w.Write(_singlelineSeparator)
		}
		io.WriteString(w, item.Error())
	}
}

func (merr *multiError) writeMultiline(w io.Writer) {
	w.Write(_multilinePrefix)
	for _, item := range merr.errors {
		w.Write(_multilineSeparator)
		writePrefixLine(w, _multilineIndent, fmt.Sprintf("%+v", item))
	}
}

// Writes s to the writer with the given prefix added before each line after
// the first.
func writePrefixLine(w io.Writer, prefix []byte, s string) {
	first := true
	for len(s) > 0 {
		if first {
			first = false
		} else {
			w.Write(prefix)
		}

		idx := strings.IndexByte(s, '\n')
		if idx < 0 {
			idx = len(s) - 1
		}

		io.WriteString(w, s[:idx+1])
		s = s[idx+1:]
	}
}

type inspectResult struct {
	// Number of top-level non-nil errors
	Count int

	// Total number of errors including multiErrors
	Capacity int

	// Index of the first non-nil error in the list. Value is meaningless if
	// Count is zero.
	FirstErrorIdx int

	// Whether the list contains at least one multiError
	ContainsMultiError bool
}

// Inspects the given slice of errors so that we can efficiently allocate
// space for it.
func inspect(errors []error) (res inspectResult) {
	first := true
	for i, err := range errors {
		if err == nil {
			continue
		}

		res.Count++
		if first {
			first = false
			res.FirstErrorIdx = i
		}

		if merr, ok := err.(*multiError); ok {
			res.Capacity += len(merr.errors)
			res.ContainsMultiError = true
		} else {
			res.Capacity++
		}
	}
	return
}

// fromSlice converts the given list of errors into a single error.
func fromSlice(errors []error) error {
	res := inspect(errors)
	switch res.Count {
	case 0:
		return nil
	case 1:
		// only one non-nil entry
		return errors[res.FirstErrorIdx]
	case len(errors):
		if !res.ContainsMultiError {
			// already flat
			return &multiError{errors: errors}
		}
	}

	nonNilErrs := make([]error, 0, res.Capacity)
	for _, err := range errors[res.FirstErrorIdx:] {
		if err == nil {
			continue
		}

		if nested, ok := err.(*multiError); ok {
			nonNilErrs = append(nonNilErrs, nested.errors...)
		} else {
			nonNilErrs = append(nonNilErrs, err)
		}
	}

	return &multiError{errors: nonNilErrs}
}

// Combine combines the passed errors into a single error.
//
// If zero arguments were passed or if all items are nil, a nil error is
// returned.
//
// 	Combine(nil, nil)  // == nil
//
// If only a single error was passed, it is returned as-is.
//
// 	Combine(err)  // == err
//
// Combine skips over nil arguments so this function may be used to combine
// together errors from operations that fail independently of each other.
//
// 	multierr.Combine(
// 		reader.Close(),
// 		writer.Close(),
// 		pipe.Close(),
// 	)
//
// If any of the passed errors is a multierr error, it will be flattened along
// with the other errors.
//
// 	multierr.Combine(multierr.Combine(err1, err2), err3)
// 	// is the same as
// 	multierr.Combine(err1, err2, err3)
//
// The returned error formats into a readable multi-line error message if
// formatted with %+v.
//
// 	fmt.Sprintf("%+v", multierr.Combine(err1, err2))
func Combine(errors ...error) error {
	return fromSlice(errors)
}

// Append appends the given errors together. Either value may be nil.
//
// This function is a specialization of Combine for the common case where
// there are only two errors.
//
// 	err = multierr.Append(reader.Close(), writer.Close())
//
// The following pattern may also be used to record failure of deferred
// operations without losing information about the original error.
//
// 	func doSomething(..) (err error) {
// 		f := acquireResource()
// 		defer func() {
// 			err = multierr.Append(err, f.Close())
// 		}()
func Append(left error, right error) error {
	switch {
	case left == nil:
		return right
	case right == nil:
		return left
	}

	if _, ok := right.(*multiError); !ok {
		if l, ok := left.(*multiError); ok && !l.copyNeeded.Swap(true) {
			// Common case where the error on the left is constantly being
			// appended to.
			errs := append(l.errors, right)
			return &multiError{errors: errs}
		} else if !ok {
			// Both errors are single errors.
			return &multiError{errors: []error{left, right}}
		}
	}

	// Either right or both, left and right, are multiErrors. Rely on usual
	// expensive logic.
	errors := [2]error{left, right}
	return fromSlice(errors[0:])
}
