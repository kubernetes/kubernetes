package fsnoder

import (
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"

	. "gopkg.in/check.v1"
)

type FSNoderSuite struct{}

var _ = Suite(&FSNoderSuite{})

func check(c *C, input string, expected *dir) {
	obtained, err := New(input)
	c.Assert(err, IsNil, Commentf("input = %s", input))

	comment := Commentf("\n   input = %s\n"+
		"expected = %s\nobtained = %s",
		input, expected, obtained)
	c.Assert(obtained.Hash(), DeepEquals, expected.Hash(), comment)
}

func (s *FSNoderSuite) TestNoDataFails(c *C) {
	_, err := New("")
	c.Assert(err, Not(IsNil))

	_, err = New(" 	") // SPC + TAB
	c.Assert(err, Not(IsNil))
}

func (s *FSNoderSuite) TestUnnamedRootFailsIfNotRoot(c *C) {
	_, err := decodeDir([]byte("()"), false)
	c.Assert(err, Not(IsNil))
}

func (s *FSNoderSuite) TestUnnamedInnerFails(c *C) {
	_, err := New("(())")
	c.Assert(err, Not(IsNil))
	_, err = New("((a<>))")
	c.Assert(err, Not(IsNil))
}

func (s *FSNoderSuite) TestMalformedFile(c *C) {
	_, err := New("(4<>)")
	c.Assert(err, Not(IsNil))
	_, err = New("(4<1>)")
	c.Assert(err, Not(IsNil))
	_, err = New("(4?1>)")
	c.Assert(err, Not(IsNil))
	_, err = New("(4<a>)")
	c.Assert(err, Not(IsNil))
	_, err = New("(4<a?)")
	c.Assert(err, Not(IsNil))

	_, err = decodeFile([]byte("a?1>"))
	c.Assert(err, Not(IsNil))
	_, err = decodeFile([]byte("a<a>"))
	c.Assert(err, Not(IsNil))
	_, err = decodeFile([]byte("a<1?"))
	c.Assert(err, Not(IsNil))

	_, err = decodeFile([]byte("a?>"))
	c.Assert(err, Not(IsNil))
	_, err = decodeFile([]byte("1<>"))
	c.Assert(err, Not(IsNil))
	_, err = decodeFile([]byte("a<?"))
	c.Assert(err, Not(IsNil))
}

func (s *FSNoderSuite) TestMalformedRootFails(c *C) {
	_, err := New(")")
	c.Assert(err, Not(IsNil))
	_, err = New("(")
	c.Assert(err, Not(IsNil))
	_, err = New("(a<>")
	c.Assert(err, Not(IsNil))
	_, err = New("a<>")
	c.Assert(err, Not(IsNil))
}

func (s *FSNoderSuite) TestUnnamedEmptyRoot(c *C) {
	input := "()"

	expected, err := newDir("", nil)
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestNamedEmptyRoot(c *C) {
	input := "a()"

	expected, err := newDir("a", nil)
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestEmptyFile(c *C) {
	input := "(a<>)"

	a1, err := newFile("a", "")
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{a1})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestNonEmptyFile(c *C) {
	input := "(a<1>)"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{a1})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestTwoFilesSameContents(c *C) {
	input := "(b<1> a<1>)"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	b1, err := newFile("b", "1")
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{a1, b1})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestTwoFilesDifferentContents(c *C) {
	input := "(b<2> a<1>)"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	b2, err := newFile("b", "2")
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{a1, b2})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestManyFiles(c *C) {
	input := "(e<1> b<2> a<1> c<1> d<3> f<4>)"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	b2, err := newFile("b", "2")
	c.Assert(err, IsNil)
	c1, err := newFile("c", "1")
	c.Assert(err, IsNil)
	d3, err := newFile("d", "3")
	c.Assert(err, IsNil)
	e1, err := newFile("e", "1")
	c.Assert(err, IsNil)
	f4, err := newFile("f", "4")
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{e1, b2, a1, c1, d3, f4})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestEmptyDir(c *C) {
	input := "(A())"

	A, err := newDir("A", nil)
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithEmtpyFile(c *C) {
	input := "(A(a<>))"

	a, err := newFile("a", "")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{a})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithEmtpyFileSameName(c *C) {
	input := "(A(A<>))"

	f, err := newFile("A", "")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{f})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithFileLongContents(c *C) {
	input := "(A(a<12>))"

	a1, err := newFile("a", "12")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{a1})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithFileLongName(c *C) {
	input := "(A(abc<12>))"

	a1, err := newFile("abc", "12")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{a1})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithFile(c *C) {
	input := "(A(a<1>))"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{a1})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithEmptyDirSameName(c *C) {
	input := "(A(A()))"

	A2, err := newDir("A", nil)
	c.Assert(err, IsNil)
	A1, err := newDir("A", []noder.Noder{A2})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A1})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithEmptyDir(c *C) {
	input := "(A(B()))"

	B, err := newDir("B", nil)
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{B})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestDirWithTwoFiles(c *C) {
	input := "(A(a<1> b<2>))"

	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	b2, err := newFile("b", "2")
	c.Assert(err, IsNil)
	A, err := newDir("A", []noder.Noder{b2, a1})
	c.Assert(err, IsNil)
	expected, err := newDir("", []noder.Noder{A})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestCrazy(c *C) {
	//           ""
	//            |
	//   -------------------------
	//   |    |      |      |    |
	//  a1    B     c1     d2    E
	//        |                  |
	//   -------------           E
	//   |   |   |   |           |
	//   A   B   X   c1          E
	//           |               |
	//          a1               e1
	input := "(d<2> b(c<1> b() a() x(a<1>)) a<1> c<1> e(e(e(e<1>))))"

	e1, err := newFile("e", "1")
	c.Assert(err, IsNil)
	E, err := newDir("e", []noder.Noder{e1})
	c.Assert(err, IsNil)
	E, err = newDir("e", []noder.Noder{E})
	c.Assert(err, IsNil)
	E, err = newDir("e", []noder.Noder{E})
	c.Assert(err, IsNil)

	A, err := newDir("a", nil)
	c.Assert(err, IsNil)
	B, err := newDir("b", nil)
	c.Assert(err, IsNil)
	a1, err := newFile("a", "1")
	c.Assert(err, IsNil)
	X, err := newDir("x", []noder.Noder{a1})
	c.Assert(err, IsNil)
	c1, err := newFile("c", "1")
	c.Assert(err, IsNil)
	B, err = newDir("b", []noder.Noder{c1, B, X, A})
	c.Assert(err, IsNil)

	a1, err = newFile("a", "1")
	c.Assert(err, IsNil)
	c1, err = newFile("c", "1")
	c.Assert(err, IsNil)
	d2, err := newFile("d", "2")
	c.Assert(err, IsNil)

	expected, err := newDir("", []noder.Noder{a1, d2, E, B, c1})
	c.Assert(err, IsNil)

	check(c, input, expected)
}

func (s *FSNoderSuite) TestHashEqual(c *C) {
	input1 := "(A(a<1> b<2>))"
	input2 := "(A(a<1> b<2>))"
	input3 := "(A(a<> b<2>))"

	t1, err := New(input1)
	c.Assert(err, IsNil)
	t2, err := New(input2)
	c.Assert(err, IsNil)
	t3, err := New(input3)
	c.Assert(err, IsNil)

	c.Assert(HashEqual(t1, t2), Equals, true)
	c.Assert(HashEqual(t2, t1), Equals, true)

	c.Assert(HashEqual(t2, t3), Equals, false)
	c.Assert(HashEqual(t3, t2), Equals, false)

	c.Assert(HashEqual(t3, t1), Equals, false)
	c.Assert(HashEqual(t1, t3), Equals, false)
}
