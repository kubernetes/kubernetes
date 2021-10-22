// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testprotos

func Equal(x, y Message) bool {
	if x == nil || y == nil {
		return x == nil && y == nil
	}
	return x.String() == y.String()
}

type Message interface {
	Proto()
	String() string
}

type proto interface {
	Proto()
}

type notComparable struct {
	unexportedField func()
}

type Stringer struct{ X string }

func (s *Stringer) String() string { return s.X }

// Project1 protocol buffers
type (
	Eagle_States         int
	Eagle_MissingCalls   int
	Dreamer_States       int
	Dreamer_MissingCalls int
	Slap_States          int
	Goat_States          int
	Donkey_States        int
	SummerType           int

	Eagle struct {
		proto
		notComparable
		Stringer
	}
	Dreamer struct {
		proto
		notComparable
		Stringer
	}
	Slap struct {
		proto
		notComparable
		Stringer
	}
	Goat struct {
		proto
		notComparable
		Stringer
	}
	Donkey struct {
		proto
		notComparable
		Stringer
	}
)

// Project2 protocol buffers
type (
	Germ struct {
		proto
		notComparable
		Stringer
	}
	Dish struct {
		proto
		notComparable
		Stringer
	}
)

// Project3 protocol buffers
type (
	Dirt struct {
		proto
		notComparable
		Stringer
	}
	Wizard struct {
		proto
		notComparable
		Stringer
	}
	Sadistic struct {
		proto
		notComparable
		Stringer
	}
)

// Project4 protocol buffers
type (
	HoneyStatus int
	PoisonType  int
	MetaData    struct {
		proto
		notComparable
		Stringer
	}
	Restrictions struct {
		proto
		notComparable
		Stringer
	}
)
