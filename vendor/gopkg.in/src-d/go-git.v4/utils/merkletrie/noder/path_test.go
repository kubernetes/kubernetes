package noder

import (
	"golang.org/x/text/unicode/norm"
	. "gopkg.in/check.v1"
)

type PathSuite struct{}

var _ = Suite(&PathSuite{})

func (s *PathSuite) TestShortFile(c *C) {
	f := &noderMock{
		name:  "1",
		isDir: false,
	}
	p := Path([]Noder{f})
	c.Assert(p.String(), Equals, "1")
}

func (s *PathSuite) TestShortDir(c *C) {
	d := &noderMock{
		name:     "1",
		isDir:    true,
		children: NoChildren,
	}
	p := Path([]Noder{d})
	c.Assert(p.String(), Equals, "1")
}

func (s *PathSuite) TestLongFile(c *C) {
	n3 := &noderMock{
		name:  "3",
		isDir: false,
	}
	n2 := &noderMock{
		name:     "2",
		isDir:    true,
		children: []Noder{n3},
	}
	n1 := &noderMock{
		name:     "1",
		isDir:    true,
		children: []Noder{n2},
	}
	p := Path([]Noder{n1, n2, n3})
	c.Assert(p.String(), Equals, "1/2/3")
}

func (s *PathSuite) TestLongDir(c *C) {
	n3 := &noderMock{
		name:     "3",
		isDir:    true,
		children: NoChildren,
	}
	n2 := &noderMock{
		name:     "2",
		isDir:    true,
		children: []Noder{n3},
	}
	n1 := &noderMock{
		name:     "1",
		isDir:    true,
		children: []Noder{n2},
	}
	p := Path([]Noder{n1, n2, n3})
	c.Assert(p.String(), Equals, "1/2/3")
}

func (s *PathSuite) TestCompareDepth1(c *C) {
	p1 := Path([]Noder{&noderMock{name: "a"}})
	p2 := Path([]Noder{&noderMock{name: "b"}})
	c.Assert(p1.Compare(p2), Equals, -1)
	c.Assert(p2.Compare(p1), Equals, 1)

	p1 = Path([]Noder{&noderMock{name: "a"}})
	p2 = Path([]Noder{&noderMock{name: "a"}})
	c.Assert(p1.Compare(p2), Equals, 0)
	c.Assert(p2.Compare(p1), Equals, 0)

	p1 = Path([]Noder{&noderMock{name: "a.go"}})
	p2 = Path([]Noder{&noderMock{name: "a"}})
	c.Assert(p1.Compare(p2), Equals, 1)
	c.Assert(p2.Compare(p1), Equals, -1)
}

func (s *PathSuite) TestCompareDepth2(c *C) {
	p1 := Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "b"},
	})
	p2 := Path([]Noder{
		&noderMock{name: "b"},
		&noderMock{name: "a"},
	})
	c.Assert(p1.Compare(p2), Equals, -1)
	c.Assert(p2.Compare(p1), Equals, 1)

	p1 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "b"},
	})
	p2 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "b"},
	})
	c.Assert(p1.Compare(p2), Equals, 0)
	c.Assert(p2.Compare(p1), Equals, 0)

	p1 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "b"},
	})
	p2 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "a"},
	})
	c.Assert(p1.Compare(p2), Equals, 1)
	c.Assert(p2.Compare(p1), Equals, -1)
}

func (s *PathSuite) TestCompareMixedDepths(c *C) {
	p1 := Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "b"},
	})
	p2 := Path([]Noder{&noderMock{name: "b"}})
	c.Assert(p1.Compare(p2), Equals, -1)
	c.Assert(p2.Compare(p1), Equals, 1)

	p1 = Path([]Noder{
		&noderMock{name: "b"},
		&noderMock{name: "b"},
	})
	p2 = Path([]Noder{&noderMock{name: "b"}})
	c.Assert(p1.Compare(p2), Equals, 1)
	c.Assert(p2.Compare(p1), Equals, -1)

	p1 = Path([]Noder{&noderMock{name: "a.go"}})
	p2 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "a.go"},
	})
	c.Assert(p1.Compare(p2), Equals, 1)
	c.Assert(p2.Compare(p1), Equals, -1)

	p1 = Path([]Noder{&noderMock{name: "b.go"}})
	p2 = Path([]Noder{
		&noderMock{name: "a"},
		&noderMock{name: "a.go"},
	})
	c.Assert(p1.Compare(p2), Equals, 1)
	c.Assert(p2.Compare(p1), Equals, -1)
}

func (s *PathSuite) TestCompareNormalization(c *C) {
	p1 := Path([]Noder{&noderMock{name: norm.Form(norm.NFKC).String("페")}})
	p2 := Path([]Noder{&noderMock{name: norm.Form(norm.NFKD).String("페")}})
	c.Assert(p1.Compare(p2), Equals, 0)
	c.Assert(p2.Compare(p1), Equals, 0)
}
