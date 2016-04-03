package foo

type wobble struct {
	Some      string
	Val       string
	Inception *wobble
}

type Fooer interface{}

type Fooer2 interface {
	Foo()
}

type Fooer3 interface {
	Foo()
	Bar(a string)
	Baz(a string) (err error)
	Qux(a, b string) (val string, err error)
	Wobble() (w *wobble)
	Wiggle() (w wobble)
}

type Fooer4 interface {
	Foo() error
}

type Bar interface {
	Boo(a string, b string) (s string, err error)
}

type Fooer5 interface {
	Foo()
	Bar
}
