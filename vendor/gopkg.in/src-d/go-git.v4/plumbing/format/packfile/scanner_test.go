package packfile

import (
	"bytes"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type ScannerSuite struct {
	fixtures.Suite
}

var _ = Suite(&ScannerSuite{})

func (s *ScannerSuite) TestHeader(c *C) {
	r := fixtures.Basic().One().Packfile()
	p := NewScanner(r)

	version, objects, err := p.Header()
	c.Assert(err, IsNil)
	c.Assert(version, Equals, VersionSupported)
	c.Assert(objects, Equals, uint32(31))
}

func (s *ScannerSuite) TestNextObjectHeaderWithoutHeader(c *C) {
	r := fixtures.Basic().One().Packfile()
	p := NewScanner(r)

	h, err := p.NextObjectHeader()
	c.Assert(err, IsNil)
	c.Assert(h, DeepEquals, &expectedHeadersOFS[0])

	version, objects, err := p.Header()
	c.Assert(err, IsNil)
	c.Assert(version, Equals, VersionSupported)
	c.Assert(objects, Equals, uint32(31))
}

func (s *ScannerSuite) TestNextObjectHeaderREFDelta(c *C) {
	s.testNextObjectHeader(c, "ref-delta", expectedHeadersREF)
}

func (s *ScannerSuite) TestNextObjectHeaderOFSDelta(c *C) {
	s.testNextObjectHeader(c, "ofs-delta", expectedHeadersOFS)
}

func (s *ScannerSuite) testNextObjectHeader(c *C, tag string, expected []ObjectHeader) {
	r := fixtures.Basic().ByTag(tag).One().Packfile()
	p := NewScanner(r)

	_, objects, err := p.Header()
	c.Assert(err, IsNil)

	for i := 0; i < int(objects); i++ {
		h, err := p.NextObjectHeader()
		c.Assert(err, IsNil)
		c.Assert(*h, DeepEquals, expected[i])

		buf := bytes.NewBuffer(nil)
		n, _, err := p.NextObject(buf)
		c.Assert(err, IsNil)
		c.Assert(n, Equals, h.Length)
	}

	n, err := p.Checksum()
	c.Assert(err, IsNil)
	c.Assert(n, HasLen, 20)
}

func (s *ScannerSuite) TestNextObjectHeaderWithOutReadObject(c *C) {
	f := fixtures.Basic().ByTag("ref-delta").One()
	r := f.Packfile()
	p := NewScanner(r)

	_, objects, err := p.Header()
	c.Assert(err, IsNil)

	for i := 0; i < int(objects); i++ {
		h, _ := p.NextObjectHeader()
		c.Assert(err, IsNil)
		c.Assert(*h, DeepEquals, expectedHeadersREF[i])
	}

	err = p.discardObjectIfNeeded()
	c.Assert(err, IsNil)

	n, err := p.Checksum()
	c.Assert(err, IsNil)
	c.Assert(n, Equals, f.PackfileHash)
}

func (s *ScannerSuite) TestNextObjectHeaderWithOutReadObjectNonSeekable(c *C) {
	f := fixtures.Basic().ByTag("ref-delta").One()
	r := io.MultiReader(f.Packfile())
	p := NewScanner(r)

	_, objects, err := p.Header()
	c.Assert(err, IsNil)

	for i := 0; i < int(objects); i++ {
		h, _ := p.NextObjectHeader()
		c.Assert(err, IsNil)
		c.Assert(*h, DeepEquals, expectedHeadersREF[i])
	}

	err = p.discardObjectIfNeeded()
	c.Assert(err, IsNil)

	n, err := p.Checksum()
	c.Assert(err, IsNil)
	c.Assert(n, Equals, f.PackfileHash)
}

var expectedHeadersOFS = []ObjectHeader{
	{Type: plumbing.CommitObject, Offset: 12, Length: 254},
	{Type: plumbing.OFSDeltaObject, Offset: 186, Length: 93, OffsetReference: 12},
	{Type: plumbing.CommitObject, Offset: 286, Length: 242},
	{Type: plumbing.CommitObject, Offset: 449, Length: 242},
	{Type: plumbing.CommitObject, Offset: 615, Length: 333},
	{Type: plumbing.CommitObject, Offset: 838, Length: 332},
	{Type: plumbing.CommitObject, Offset: 1063, Length: 244},
	{Type: plumbing.CommitObject, Offset: 1230, Length: 243},
	{Type: plumbing.CommitObject, Offset: 1392, Length: 187},
	{Type: plumbing.BlobObject, Offset: 1524, Length: 189},
	{Type: plumbing.BlobObject, Offset: 1685, Length: 18},
	{Type: plumbing.BlobObject, Offset: 1713, Length: 1072},
	{Type: plumbing.BlobObject, Offset: 2351, Length: 76110},
	{Type: plumbing.BlobObject, Offset: 78050, Length: 2780},
	{Type: plumbing.BlobObject, Offset: 78882, Length: 217848},
	{Type: plumbing.BlobObject, Offset: 80725, Length: 706},
	{Type: plumbing.BlobObject, Offset: 80998, Length: 11488},
	{Type: plumbing.BlobObject, Offset: 84032, Length: 78},
	{Type: plumbing.TreeObject, Offset: 84115, Length: 272},
	{Type: plumbing.OFSDeltaObject, Offset: 84375, Length: 43, OffsetReference: 84115},
	{Type: plumbing.TreeObject, Offset: 84430, Length: 38},
	{Type: plumbing.TreeObject, Offset: 84479, Length: 75},
	{Type: plumbing.TreeObject, Offset: 84559, Length: 38},
	{Type: plumbing.TreeObject, Offset: 84608, Length: 34},
	{Type: plumbing.BlobObject, Offset: 84653, Length: 9},
	{Type: plumbing.OFSDeltaObject, Offset: 84671, Length: 6, OffsetReference: 84375},
	{Type: plumbing.OFSDeltaObject, Offset: 84688, Length: 9, OffsetReference: 84375},
	{Type: plumbing.OFSDeltaObject, Offset: 84708, Length: 6, OffsetReference: 84375},
	{Type: plumbing.OFSDeltaObject, Offset: 84725, Length: 5, OffsetReference: 84115},
	{Type: plumbing.OFSDeltaObject, Offset: 84741, Length: 8, OffsetReference: 84375},
	{Type: plumbing.OFSDeltaObject, Offset: 84760, Length: 4, OffsetReference: 84741},
}

var expectedHeadersREF = []ObjectHeader{
	{Type: plumbing.CommitObject, Offset: 12, Length: 254},
	{Type: plumbing.REFDeltaObject, Offset: 186, Length: 93,
		Reference: plumbing.NewHash("e8d3ffab552895c19b9fcf7aa264d277cde33881")},
	{Type: plumbing.CommitObject, Offset: 304, Length: 242},
	{Type: plumbing.CommitObject, Offset: 467, Length: 242},
	{Type: plumbing.CommitObject, Offset: 633, Length: 333},
	{Type: plumbing.CommitObject, Offset: 856, Length: 332},
	{Type: plumbing.CommitObject, Offset: 1081, Length: 243},
	{Type: plumbing.CommitObject, Offset: 1243, Length: 244},
	{Type: plumbing.CommitObject, Offset: 1410, Length: 187},
	{Type: plumbing.BlobObject, Offset: 1542, Length: 189},
	{Type: plumbing.BlobObject, Offset: 1703, Length: 18},
	{Type: plumbing.BlobObject, Offset: 1731, Length: 1072},
	{Type: plumbing.BlobObject, Offset: 2369, Length: 76110},
	{Type: plumbing.TreeObject, Offset: 78068, Length: 38},
	{Type: plumbing.BlobObject, Offset: 78117, Length: 2780},
	{Type: plumbing.TreeObject, Offset: 79049, Length: 75},
	{Type: plumbing.BlobObject, Offset: 79129, Length: 217848},
	{Type: plumbing.BlobObject, Offset: 80972, Length: 706},
	{Type: plumbing.TreeObject, Offset: 81265, Length: 38},
	{Type: plumbing.BlobObject, Offset: 81314, Length: 11488},
	{Type: plumbing.TreeObject, Offset: 84752, Length: 34},
	{Type: plumbing.BlobObject, Offset: 84797, Length: 78},
	{Type: plumbing.TreeObject, Offset: 84880, Length: 271},
	{Type: plumbing.REFDeltaObject, Offset: 85141, Length: 6,
		Reference: plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c")},
	{Type: plumbing.REFDeltaObject, Offset: 85176, Length: 37,
		Reference: plumbing.NewHash("fb72698cab7617ac416264415f13224dfd7a165e")},
	{Type: plumbing.BlobObject, Offset: 85244, Length: 9},
	{Type: plumbing.REFDeltaObject, Offset: 85262, Length: 9,
		Reference: plumbing.NewHash("fb72698cab7617ac416264415f13224dfd7a165e")},
	{Type: plumbing.REFDeltaObject, Offset: 85300, Length: 6,
		Reference: plumbing.NewHash("fb72698cab7617ac416264415f13224dfd7a165e")},
	{Type: plumbing.TreeObject, Offset: 85335, Length: 110},
	{Type: plumbing.REFDeltaObject, Offset: 85448, Length: 8,
		Reference: plumbing.NewHash("eba74343e2f15d62adedfd8c883ee0262b5c8021")},
	{Type: plumbing.TreeObject, Offset: 85485, Length: 73},
}
