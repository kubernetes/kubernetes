package foo

import (
	"fmt"

	aliasedio "io"

	"github.com/docker/docker/pkg/plugins/pluginrpc-gen/fixtures/otherfixture"
)

var (
	errFakeImport = fmt.Errorf("just to import fmt for imports tests")
)

type wobble struct {
	Some      string
	Val       string
	Inception *wobble
}

// Fooer is an empty interface used for tests.
type Fooer interface{}

// Fooer2 is an interface used for tests.
type Fooer2 interface {
	Foo()
}

// Fooer3 is an interface used for tests.
type Fooer3 interface {
	Foo()
	Bar(a string)
	Baz(a string) (err error)
	Qux(a, b string) (val string, err error)
	Wobble() (w *wobble)
	Wiggle() (w wobble)
	WiggleWobble(a []*wobble, b []wobble, c map[string]*wobble, d map[*wobble]wobble, e map[string][]wobble, f []*otherfixture.Spaceship) (g map[*wobble]wobble, h [][]*wobble, i otherfixture.Spaceship, j *otherfixture.Spaceship, k map[*otherfixture.Spaceship]otherfixture.Spaceship, l []otherfixture.Spaceship)
}

// Fooer4 is an interface used for tests.
type Fooer4 interface {
	Foo() error
}

// Bar is an interface used for tests.
type Bar interface {
	Boo(a string, b string) (s string, err error)
}

// Fooer5 is an interface used for tests.
type Fooer5 interface {
	Foo()
	Bar
}

// Fooer6 is an interface used for tests.
type Fooer6 interface {
	Foo(a otherfixture.Spaceship)
}

// Fooer7 is an interface used for tests.
type Fooer7 interface {
	Foo(a *otherfixture.Spaceship)
}

// Fooer8 is an interface used for tests.
type Fooer8 interface {
	Foo(a map[string]otherfixture.Spaceship)
}

// Fooer9 is an interface used for tests.
type Fooer9 interface {
	Foo(a map[string]*otherfixture.Spaceship)
}

// Fooer10 is an interface used for tests.
type Fooer10 interface {
	Foo(a []otherfixture.Spaceship)
}

// Fooer11 is an interface used for tests.
type Fooer11 interface {
	Foo(a []*otherfixture.Spaceship)
}

// Fooer12 is an interface used for tests.
type Fooer12 interface {
	Foo(a aliasedio.Reader)
}
