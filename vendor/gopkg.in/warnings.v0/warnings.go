// Package warnings implements error handling with non-fatal errors (warnings).
//
// A recurring pattern in Go programming is the following:
//
//  func myfunc(params) error {
//      if err := doSomething(...); err != nil {
//          return err
//      }
//      if err := doSomethingElse(...); err != nil {
//          return err
//      }
//      if ok := doAnotherThing(...); !ok {
//          return errors.New("my error")
//      }
//      ...
//      return nil
//  }
//
// This pattern allows interrupting the flow on any received error. But what if
// there are errors that should be noted but still not fatal, for which the flow
// should not be interrupted? Implementing such logic at each if statement would
// make the code complex and the flow much harder to follow.
//
// Package warnings provides the Collector type and a clean and simple pattern
// for achieving such logic. The Collector takes care of deciding when to break
// the flow and when to continue, collecting any non-fatal errors (warnings)
// along the way. The only requirement is that fatal and non-fatal errors can be
// distinguished programmatically; that is a function such as
//
//  IsFatal(error) bool
//
// must be implemented. The following is an example of what the above snippet
// could look like using the warnings package:
//
//  import "gopkg.in/warnings.v0"
//
//  func isFatal(err error) bool {
//      _, ok := err.(WarningType)
//      return !ok
//  }
//
//  func myfunc(params) error {
//      c := warnings.NewCollector(isFatal)
//      c.FatalWithWarnings = true
//      if err := c.Collect(doSomething()); err != nil {
//          return err
//      }
//      if err := c.Collect(doSomethingElse(...)); err != nil {
//          return err
//      }
//      if ok := doAnotherThing(...); !ok {
//          if err := c.Collect(errors.New("my error")); err != nil {
//              return err
//          }
//      }
//      ...
//      return c.Done()
//  }
//
// Rules for using warnings
//
//  - ensure that warnings are programmatically distinguishable from fatal
//    errors (i.e. implement an isFatal function and any necessary error types)
//  - ensure that there is a single Collector instance for a call of each
//    exported function
//  - ensure that all errors (fatal or warning) are fed through Collect
//  - ensure that every time an error is returned, it is one returned by a
//    Collector (from Collect or Done)
//  - ensure that Collect is never called after Done
//
// TODO
//
//  - optionally limit the number of warnings (e.g. stop after 20 warnings) (?)
//  - consider interaction with contexts
//  - go vet-style invocations verifier
//  - semi-automatic code converter
//
package warnings

import (
	"bytes"
	"fmt"
)

// List holds a collection of warnings and optionally one fatal error.
type List struct {
	Warnings []error
	Fatal    error
}

// Error implements the error interface.
func (l List) Error() string {
	b := bytes.NewBuffer(nil)
	if l.Fatal != nil {
		fmt.Fprintln(b, "fatal:")
		fmt.Fprintln(b, l.Fatal)
	}
	switch len(l.Warnings) {
	case 0:
	// nop
	case 1:
		fmt.Fprintln(b, "warning:")
	default:
		fmt.Fprintln(b, "warnings:")
	}
	for _, err := range l.Warnings {
		fmt.Fprintln(b, err)
	}
	return b.String()
}

// A Collector collects errors up to the first fatal error.
type Collector struct {
	// IsFatal distinguishes between warnings and fatal errors.
	IsFatal func(error) bool
	// FatalWithWarnings set to true means that a fatal error is returned as
	// a List together with all warnings so far. The default behavior is to
	// only return the fatal error and discard any warnings that have been
	// collected.
	FatalWithWarnings bool

	l    List
	done bool
}

// NewCollector returns a new Collector; it uses isFatal to distinguish between
// warnings and fatal errors.
func NewCollector(isFatal func(error) bool) *Collector {
	return &Collector{IsFatal: isFatal}
}

// Collect collects a single error (warning or fatal). It returns nil if
// collection can continue (only warnings so far), or otherwise the errors
// collected. Collect mustn't be called after the first fatal error or after
// Done has been called.
func (c *Collector) Collect(err error) error {
	if c.done {
		panic("warnings.Collector already done")
	}
	if err == nil {
		return nil
	}
	if c.IsFatal(err) {
		c.done = true
		c.l.Fatal = err
	} else {
		c.l.Warnings = append(c.l.Warnings, err)
	}
	if c.l.Fatal != nil {
		return c.erorr()
	}
	return nil
}

// Done ends collection and returns the collected error(s).
func (c *Collector) Done() error {
	c.done = true
	return c.erorr()
}

func (c *Collector) erorr() error {
	if !c.FatalWithWarnings && c.l.Fatal != nil {
		return c.l.Fatal
	}
	if c.l.Fatal == nil && len(c.l.Warnings) == 0 {
		return nil
	}
	// Note that a single warning is also returned as a List. This is to make it
	// easier to determine fatal-ness of the returned error.
	return c.l
}

// FatalOnly returns the fatal error, if any, **in an error returned by a
// Collector**. It returns nil if and only if err is nil or err is a List
// with err.Fatal == nil.
func FatalOnly(err error) error {
	l, ok := err.(List)
	if !ok {
		return err
	}
	return l.Fatal
}

// WarningsOnly returns the warnings **in an error returned by a Collector**.
func WarningsOnly(err error) []error {
	l, ok := err.(List)
	if !ok {
		return nil
	}
	return l.Warnings
}
