// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	goflag "flag"
	"fmt"
	"reflect"
	"strings"
)

var _ = fmt.Print

// flagValueWrapper implements pflag.Value around a flag.Value.  The main
// difference here is the addition of the Type method that returns a string
// name of the type.  As this is generally unknown, we approximate that with
// reflection.
type flagValueWrapper struct {
	inner    goflag.Value
	flagType string
}

// We are just copying the boolFlag interface out of goflag as that is what
// they use to decide if a flag should get "true" when no arg is given.
type goBoolFlag interface {
	goflag.Value
	IsBoolFlag() bool
}

func wrapFlagValue(v goflag.Value) Value {
	// If the flag.Value happens to also be a pflag.Value, just use it directly.
	if pv, ok := v.(Value); ok {
		return pv
	}

	pv := &flagValueWrapper{
		inner: v,
	}

	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Interface || t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	pv.flagType = strings.TrimSuffix(t.Name(), "Value")
	return pv
}

func (v *flagValueWrapper) String() string {
	return v.inner.String()
}

func (v *flagValueWrapper) Set(s string) error {
	return v.inner.Set(s)
}

func (v *flagValueWrapper) Type() string {
	return v.flagType
}

func PFlagFromGoFlag(goflag *goflag.Flag) *Flag {
	// Remember the default value as a string; it won't change.
	flag := &Flag{
		Name:  goflag.Name,
		Usage: goflag.Usage,
		Value: wrapFlagValue(goflag.Value),
		// Looks like golang flags don't set DefValue correctly  :-(
		//DefValue: goflag.DefValue,
		DefValue: goflag.Value.String(),
	}
	if fv, ok := goflag.Value.(goBoolFlag); ok && fv.IsBoolFlag() {
		flag.NoOptDefVal = "true"
	}
	return flag
}

func (f *FlagSet) AddGoFlag(goflag *goflag.Flag) {
	if f.Lookup(goflag.Name) != nil {
		return
	}
	newflag := PFlagFromGoFlag(goflag)
	f.AddFlag(newflag)
}

func (f *FlagSet) AddGoFlagSet(newSet *goflag.FlagSet) {
	if newSet == nil {
		return
	}
	newSet.VisitAll(func(goflag *goflag.Flag) {
		f.AddGoFlag(goflag)
	})
}
