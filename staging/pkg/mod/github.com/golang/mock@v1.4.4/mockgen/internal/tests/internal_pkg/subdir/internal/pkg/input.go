package pkg

type Arg interface {
	Foo() int
}

type Intf interface {
	F() Arg
}
