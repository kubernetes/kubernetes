package faux

type Foreign interface {
	Method() Return
	Embedded
}

type Embedded interface{}

type Return interface{}
