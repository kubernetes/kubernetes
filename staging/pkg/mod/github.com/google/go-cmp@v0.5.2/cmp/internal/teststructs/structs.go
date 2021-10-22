// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package teststructs

type InterfaceA interface {
	InterfaceA()
}

type (
	StructA struct{ X string } // Equal method on value receiver
	StructB struct{ X string } // Equal method on pointer receiver
	StructC struct{ X string } // Equal method (with interface argument) on value receiver
	StructD struct{ X string } // Equal method (with interface argument) on pointer receiver
	StructE struct{ X string } // Equal method (with interface argument on value receiver) on pointer receiver
	StructF struct{ X string } // Equal method (with interface argument on pointer receiver) on value receiver

	// These embed the above types as a value.
	StructA1 struct {
		StructA
		X string
	}
	StructB1 struct {
		StructB
		X string
	}
	StructC1 struct {
		StructC
		X string
	}
	StructD1 struct {
		StructD
		X string
	}
	StructE1 struct {
		StructE
		X string
	}
	StructF1 struct {
		StructF
		X string
	}

	// These embed the above types as a pointer.
	StructA2 struct {
		*StructA
		X string
	}
	StructB2 struct {
		*StructB
		X string
	}
	StructC2 struct {
		*StructC
		X string
	}
	StructD2 struct {
		*StructD
		X string
	}
	StructE2 struct {
		*StructE
		X string
	}
	StructF2 struct {
		*StructF
		X string
	}

	StructNo struct{ X string } // Equal method (with interface argument) on non-satisfying receiver

	AssignA func() int
	AssignB struct{ A int }
	AssignC chan bool
	AssignD <-chan bool
)

func (x StructA) Equal(y StructA) bool     { return true }
func (x *StructB) Equal(y *StructB) bool   { return true }
func (x StructC) Equal(y InterfaceA) bool  { return true }
func (x StructC) InterfaceA()              {}
func (x *StructD) Equal(y InterfaceA) bool { return true }
func (x *StructD) InterfaceA()             {}
func (x *StructE) Equal(y InterfaceA) bool { return true }
func (x StructE) InterfaceA()              {}
func (x StructF) Equal(y InterfaceA) bool  { return true }
func (x *StructF) InterfaceA()             {}
func (x StructNo) Equal(y InterfaceA) bool { return true }

func (x AssignA) Equal(y func() int) bool      { return true }
func (x AssignB) Equal(y struct{ A int }) bool { return true }
func (x AssignC) Equal(y chan bool) bool       { return true }
func (x AssignD) Equal(y <-chan bool) bool     { return true }

var _ = func(
	a StructA, b StructB, c StructC, d StructD, e StructE, f StructF,
	ap *StructA, bp *StructB, cp *StructC, dp *StructD, ep *StructE, fp *StructF,
	a1 StructA1, b1 StructB1, c1 StructC1, d1 StructD1, e1 StructE1, f1 StructF1,
	a2 StructA2, b2 StructB2, c2 StructC2, d2 StructD2, e2 StructE2, f2 StructF1,
) {
	a.Equal(a)
	b.Equal(&b)
	c.Equal(c)
	d.Equal(&d)
	e.Equal(e)
	f.Equal(&f)

	ap.Equal(*ap)
	bp.Equal(bp)
	cp.Equal(*cp)
	dp.Equal(dp)
	ep.Equal(*ep)
	fp.Equal(fp)

	a1.Equal(a1.StructA)
	b1.Equal(&b1.StructB)
	c1.Equal(c1)
	d1.Equal(&d1)
	e1.Equal(e1)
	f1.Equal(&f1)

	a2.Equal(*a2.StructA)
	b2.Equal(b2.StructB)
	c2.Equal(c2)
	d2.Equal(&d2)
	e2.Equal(e2)
	f2.Equal(&f2)
}

type (
	privateStruct struct{ Public, private int }
	PublicStruct  struct{ Public, private int }
	ParentStructA struct{ privateStruct }
	ParentStructB struct{ PublicStruct }
	ParentStructC struct {
		privateStruct
		Public, private int
	}
	ParentStructD struct {
		PublicStruct
		Public, private int
	}
	ParentStructE struct {
		privateStruct
		PublicStruct
	}
	ParentStructF struct {
		privateStruct
		PublicStruct
		Public, private int
	}
	ParentStructG struct {
		*privateStruct
	}
	ParentStructH struct {
		*PublicStruct
	}
	ParentStructI struct {
		*privateStruct
		*PublicStruct
	}
	ParentStructJ struct {
		*privateStruct
		*PublicStruct
		Public  PublicStruct
		private privateStruct
	}
)

func NewParentStructG() *ParentStructG {
	return &ParentStructG{new(privateStruct)}
}
func NewParentStructH() *ParentStructH {
	return &ParentStructH{new(PublicStruct)}
}
func NewParentStructI() *ParentStructI {
	return &ParentStructI{new(privateStruct), new(PublicStruct)}
}
func NewParentStructJ() *ParentStructJ {
	return &ParentStructJ{
		privateStruct: new(privateStruct), PublicStruct: new(PublicStruct),
	}
}
func (s *privateStruct) SetPrivate(i int)              { s.private = i }
func (s *PublicStruct) SetPrivate(i int)               { s.private = i }
func (s *ParentStructC) SetPrivate(i int)              { s.private = i }
func (s *ParentStructD) SetPrivate(i int)              { s.private = i }
func (s *ParentStructF) SetPrivate(i int)              { s.private = i }
func (s *ParentStructA) PrivateStruct() *privateStruct { return &s.privateStruct }
func (s *ParentStructC) PrivateStruct() *privateStruct { return &s.privateStruct }
func (s *ParentStructE) PrivateStruct() *privateStruct { return &s.privateStruct }
func (s *ParentStructF) PrivateStruct() *privateStruct { return &s.privateStruct }
func (s *ParentStructG) PrivateStruct() *privateStruct { return s.privateStruct }
func (s *ParentStructI) PrivateStruct() *privateStruct { return s.privateStruct }
func (s *ParentStructJ) PrivateStruct() *privateStruct { return s.privateStruct }
func (s *ParentStructJ) Private() *privateStruct       { return &s.private }
