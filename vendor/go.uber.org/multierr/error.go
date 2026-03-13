// Copyright (c) 2017-2023 Uber Technologies, Inc.
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
// # Overview
//
// Errors can be combined with the use of the Combine function.
//
//	multierr.Combine(
//		reader.Close(),
//		writer.Close(),
//		conn.Close(),
//	)
//
// If only two errors are being combined, the Append function may be used
// instead.
//
//	err = multierr.Append(reader.Close(), writer.Close())
//
// The underlying list of errors for a returned error object may be retrieved
// with the Errors function.
//
//	errors := multierr.Errors(err)
//	if len(errors) > 0 {
//		fmt.Println("The following errors occurred:", errors)
//	}
//
// # Appending from a loop
//
// You sometimes need to append into an error from a loop.
//
//	var err error
//	for _, item := range items {
//		err = multierr.Append(err, process(item))
//	}
//
// Cases like this may require knowledge of whether an individual instance
// failed. This usually requires introduction of a new variable.
//
//	var err error
//	for _, item := range items {
//		if perr := process(item); perr != nil {
//			log.Warn("skipping item", item)
//			err = multierr.Append(err, perr)
//		}
//	}
//
// multierr includes AppendInto to simplify cases like this.
//
//	var err error
//	for _, item := range items {
//		if multierr.AppendInto(&err, process(item)) {
//			log.Warn("skipping item", item)
//		}
//	}
//
// This will append the error into the err variable, and return true if that
// individual error was non-nil.
//
// See [AppendInto] for more information.
//
// # Deferred Functions
//
// Go makes it possible to modify the return value of a function in a defer
// block if the function was using named returns. This makes it possible to
// record resource cleanup failures from deferred blocks.
//
//	func sendRequest(req Request) (err error) {
//		conn, err := openConnection()
//		if err != nil {
//			return err
//		}
//		defer func() {
//			err = multierr.Append(err, conn.Close())
//		}()
//		// ...
//	}
//
// multierr provides the Invoker type and AppendInvoke function to make cases
// like the above simpler and obviate the need for a closure. The following is
// roughly equivalent to the example above.
//
//	func sendRequest(req Request) (err error) {
//		conn, err := openConnection()
//		if err != nil {
//			return err
//		}
//		defer multierr.AppendInvoke(&err, multierr.Close(conn))
//		// ...
//	}
//
// See [AppendInvoke] and [Invoker] for more information.
//
// NOTE: If you're modifying an error from inside a defer, you MUST use a named
// return value for that function.
//
// # Advanced Usage
//
// Errors returned by Combine and Append MAY implement the following
// interface.
//
//	type errorGroup interface {
//		// Returns a slice containing the underlying list of errors.
//		//
//		// This slice MUST NOT be modified by the caller.
//		Errors() []error
//	}
//
// Note that if you need access to list of errors behind a multierr error, you
// should prefer using the Errors function. That said, if you need cheap
// read-only access to the underlying errors slice, you can attempt to cast
// the error to this interface. You MUST handle the failure case gracefully
// because errors returned by Combine and Append are not guaranteed to
// implement this interface.
//
//	var errors []error
//	group, ok := err.(errorGroup)
//	if ok {
//		errors = group.Errors()
//	} else {
//		errors = []error{err}
//	}
package multierr // import "go.uber.org/multierr"

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	// Separator for single-line error messages.
	_singlelineSeparator = []byte("; ")

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
// error is composed of. If the error is nil, a nil slice is returned.
//
//	err := multierr.Append(r.Close(), w.Close())
//	errors := multierr.Errors(err)
//
// If the error is not composed of other errors, the returned slice contains
// just the error that was passed in.
//
// Callers of this function are free to modify the returned slice.
func Errors(err error) []error {
	return extractErrors(err)
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

// Every compares every error in the given err against the given target error
// using [errors.Is], and returns true only if every comparison returned true.
func Every(err error, target error) bool {
	for _, e := range extractErrors(err) {
		if !errors.Is(e, target) {
			return false
		}
	}
	return true
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
	// Don't pay to inspect small slices.
	switch len(errors) {
	case 0:
		return nil
	case 1:
		return errors[0]
	}

	res := inspect(errors)
	switch res.Count {
	case 0:
		return nil
	case 1:
		// only one non-nil entry
		return errors[res.FirstErrorIdx]
	case len(errors):
		if !res.ContainsMultiError {
			// Error list is flat. Make a copy of it
			// Otherwise "errors" escapes to the heap
			// unconditionally for all other cases.
			// This lets us optimize for the "no errors" case.
			out := append(([]error)(nil), errors...)
			return &multiError{errors: out}
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
//	Combine(nil, nil)  // == nil
//
// If only a single error was passed, it is returned as-is.
//
//	Combine(err)  // == err
//
// Combine skips over nil arguments so this function may be used to combine
// together errors from operations that fail independently of each other.
//
//	multierr.Combine(
//		reader.Close(),
//		writer.Close(),
//		pipe.Close(),
//	)
//
// If any of the passed errors is a multierr error, it will be flattened along
// with the other errors.
//
//	multierr.Combine(multierr.Combine(err1, err2), err3)
//	// is the same as
//	multierr.Combine(err1, err2, err3)
//
// The returned error formats into a readable multi-line error message if
// formatted with %+v.
//
//	fmt.Sprintf("%+v", multierr.Combine(err1, err2))
func Combine(errors ...error) error {
	return fromSlice(errors)
}

// Append appends the given errors together. Either value may be nil.
//
// This function is a specialization of Combine for the common case where
// there are only two errors.
//
//	err = multierr.Append(reader.Close(), writer.Close())
//
// The following pattern may also be used to record failure of deferred
// operations without losing information about the original error.
//
//	func doSomething(..) (err error) {
//		f := acquireResource()
//		defer func() {
//			err = multierr.Append(err, f.Close())
//		}()
//
// Note that the variable MUST be a named return to append an error to it from
// the defer statement. See also [AppendInvoke].
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

// AppendInto appends an error into the destination of an error pointer and
// returns whether the error being appended was non-nil.
//
//	var err error
//	multierr.AppendInto(&err, r.Close())
//	multierr.AppendInto(&err, w.Close())
//
// The above is equivalent to,
//
//	err := multierr.Append(r.Close(), w.Close())
//
// As AppendInto reports whether the provided error was non-nil, it may be
// used to build a multierr error in a loop more ergonomically. For example:
//
//	var err error
//	for line := range lines {
//		var item Item
//		if multierr.AppendInto(&err, parse(line, &item)) {
//			continue
//		}
//		items = append(items, item)
//	}
//
// Compare this with a version that relies solely on Append:
//
//	var err error
//	for line := range lines {
//		var item Item
//		if parseErr := parse(line, &item); parseErr != nil {
//			err = multierr.Append(err, parseErr)
//			continue
//		}
//		items = append(items, item)
//	}
func AppendInto(into *error, err error) (errored bool) {
	if into == nil {
		// We panic if 'into' is nil. This is not documented above
		// because suggesting that the pointer must be non-nil may
		// confuse users into thinking that the error that it points
		// to must be non-nil.
		panic("misuse of multierr.AppendInto: into pointer must not be nil")
	}

	if err == nil {
		return false
	}
	*into = Append(*into, err)
	return true
}

// Invoker is an operation that may fail with an error. Use it with
// AppendInvoke to append the result of calling the function into an error.
// This allows you to conveniently defer capture of failing operations.
//
// See also, [Close] and [Invoke].
type Invoker interface {
	Invoke() error
}

// Invoke wraps a function which may fail with an error to match the Invoker
// interface. Use it to supply functions matching this signature to
// AppendInvoke.
//
// For example,
//
//	func processReader(r io.Reader) (err error) {
//		scanner := bufio.NewScanner(r)
//		defer multierr.AppendInvoke(&err, multierr.Invoke(scanner.Err))
//		for scanner.Scan() {
//			// ...
//		}
//		// ...
//	}
//
// In this example, the following line will construct the Invoker right away,
// but defer the invocation of scanner.Err() until the function returns.
//
//	defer multierr.AppendInvoke(&err, multierr.Invoke(scanner.Err))
//
// Note that the error you're appending to from the defer statement MUST be a
// named return.
type Invoke func() error

// Invoke calls the supplied function and returns its result.
func (i Invoke) Invoke() error { return i() }

// Close builds an Invoker that closes the provided io.Closer. Use it with
// AppendInvoke to close io.Closers and append their results into an error.
//
// For example,
//
//	func processFile(path string) (err error) {
//		f, err := os.Open(path)
//		if err != nil {
//			return err
//		}
//		defer multierr.AppendInvoke(&err, multierr.Close(f))
//		return processReader(f)
//	}
//
// In this example, multierr.Close will construct the Invoker right away, but
// defer the invocation of f.Close until the function returns.
//
//	defer multierr.AppendInvoke(&err, multierr.Close(f))
//
// Note that the error you're appending to from the defer statement MUST be a
// named return.
func Close(closer io.Closer) Invoker {
	return Invoke(closer.Close)
}

// AppendInvoke appends the result of calling the given Invoker into the
// provided error pointer. Use it with named returns to safely defer
// invocation of fallible operations until a function returns, and capture the
// resulting errors.
//
//	func doSomething(...) (err error) {
//		// ...
//		f, err := openFile(..)
//		if err != nil {
//			return err
//		}
//
//		// multierr will call f.Close() when this function returns and
//		// if the operation fails, its append its error into the
//		// returned error.
//		defer multierr.AppendInvoke(&err, multierr.Close(f))
//
//		scanner := bufio.NewScanner(f)
//		// Similarly, this scheduled scanner.Err to be called and
//		// inspected when the function returns and append its error
//		// into the returned error.
//		defer multierr.AppendInvoke(&err, multierr.Invoke(scanner.Err))
//
//		// ...
//	}
//
// NOTE: If used with a defer, the error variable MUST be a named return.
//
// Without defer, AppendInvoke behaves exactly like AppendInto.
//
//	err := // ...
//	multierr.AppendInvoke(&err, mutltierr.Invoke(foo))
//
//	// ...is roughly equivalent to...
//
//	err := // ...
//	multierr.AppendInto(&err, foo())
//
// The advantage of the indirection introduced by Invoker is to make it easy
// to defer the invocation of a function. Without this indirection, the
// invoked function will be evaluated at the time of the defer block rather
// than when the function returns.
//
//	// BAD: This is likely not what the caller intended. This will evaluate
//	// foo() right away and append its result into the error when the
//	// function returns.
//	defer multierr.AppendInto(&err, foo())
//
//	// GOOD: This will defer invocation of foo unutil the function returns.
//	defer multierr.AppendInvoke(&err, multierr.Invoke(foo))
//
// multierr provides a few Invoker implementations out of the box for
// convenience. See [Invoker] for more information.
func AppendInvoke(into *error, invoker Invoker) {
	AppendInto(into, invoker.Invoke())
}

// AppendFunc is a shorthand for [AppendInvoke].
// It allows using function or method value directly
// without having to wrap it into an [Invoker] interface.
//
//	func doSomething(...) (err error) {
//		w, err := startWorker(...)
//		if err != nil {
//			return err
//		}
//
//		// multierr will call w.Stop() when this function returns and
//		// if the operation fails, it appends its error into the
//		// returned error.
//		defer multierr.AppendFunc(&err, w.Stop)
//	}
func AppendFunc(into *error, fn func() error) {
	AppendInvoke(into, Invoke(fn))
}
