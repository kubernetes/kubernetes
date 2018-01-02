package idxfile_test

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"testing"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/src-d/go-git.v4/plumbing/format/idxfile"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type IdxfileSuite struct {
	fixtures.Suite
}

var _ = Suite(&IdxfileSuite{})

func (s *IdxfileSuite) TestDecode(c *C) {
	f := fixtures.Basic().One()

	d := NewDecoder(f.Idx())
	idx := &Idxfile{}
	err := d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, HasLen, 31)
	c.Assert(idx.Entries[0].Hash.String(), Equals, "1669dce138d9b841a518c64b10914d88f5e488ea")
	c.Assert(idx.Entries[0].Offset, Equals, uint64(615))
	c.Assert(idx.Entries[0].CRC32, Equals, uint32(3645019190))

	c.Assert(fmt.Sprintf("%x", idx.IdxChecksum), Equals, "fb794f1ec720b9bc8e43257451bd99c4be6fa1c9")
	c.Assert(fmt.Sprintf("%x", idx.PackfileChecksum), Equals, f.PackfileHash.String())
}

func (s *IdxfileSuite) TestDecodeCRCs(c *C) {
	f := fixtures.Basic().ByTag("ofs-delta").One()

	scanner := packfile.NewScanner(f.Packfile())
	storage := memory.NewStorage()

	pd, err := packfile.NewDecoder(scanner, storage)
	c.Assert(err, IsNil)
	_, err = pd.Decode()
	c.Assert(err, IsNil)

	i := pd.Index().ToIdxFile()
	i.Version = VersionSupported

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	_, err = e.Encode(i)
	c.Assert(err, IsNil)

	idx := &Idxfile{}

	d := NewDecoder(buf)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, DeepEquals, i.Entries)
}

func (s *IdxfileSuite) TestDecode64bitsOffsets(c *C) {
	f := bytes.NewBufferString(fixtureLarge4GB)

	idx := &Idxfile{}

	d := NewDecoder(base64.NewDecoder(base64.StdEncoding, f))
	err := d.Decode(idx)
	c.Assert(err, IsNil)

	expected := map[string]uint64{
		"303953e5aa461c203a324821bc1717f9b4fff895": 12,
		"5296768e3d9f661387ccbff18c4dea6c997fd78c": 142,
		"03fc8d58d44267274edef4585eaeeb445879d33f": 1601322837,
		"8f3ceb4ea4cb9e4a0f751795eb41c9a4f07be772": 2646996529,
		"e0d1d625010087f79c9e01ad9d8f95e1628dda02": 3452385606,
		"90eba326cdc4d1d61c5ad25224ccbf08731dd041": 3707047470,
		"bab53055add7bc35882758a922c54a874d6b1272": 5323223332,
		"1b8995f51987d8a449ca5ea4356595102dc2fbd4": 5894072943,
		"35858be9c6f5914cbe6768489c41eb6809a2bceb": 5924278919,
	}

	for _, e := range idx.Entries {
		c.Assert(expected[e.Hash.String()], Equals, e.Offset)
	}
}

func (s *IdxfileSuite) TestDecode64bitsOffsetsIdempotent(c *C) {
	f := bytes.NewBufferString(fixtureLarge4GB)

	expected := &Idxfile{}

	d := NewDecoder(base64.NewDecoder(base64.StdEncoding, f))
	err := d.Decode(expected)
	c.Assert(err, IsNil)

	buf := bytes.NewBuffer(nil)
	_, err = NewEncoder(buf).Encode(expected)
	c.Assert(err, IsNil)

	idx := &Idxfile{}
	err = NewDecoder(buf).Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, DeepEquals, expected.Entries)
}

const fixtureLarge4GB = `/3RPYwAAAAIAAAAAAAAAAAAAAAAAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEA
AAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAA
AAEAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAA
AgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAADAAAAAwAAAAMAAAADAAAAAwAAAAQAAAAE
AAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQA
AAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABQAA
AAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAA
BQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAF
AAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUA
AAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAA
AAUAAAAFAAAABQAAAAYAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAA
BwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAH
AAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAAAABwAAAAcA
AAAHAAAABwAAAAcAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAA
AAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAA
CAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAgAAAAIAAAACAAAAAkAAAAJ
AAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkA
AAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAAAAkAAAAJAAAACQAA
AAkAAAAJA/yNWNRCZydO3vRYXq7rRFh50z8biZX1GYfYpEnKXqQ1ZZUQLcL71DA5U+WqRhwgOjJI
IbwXF/m0//iVNYWL6cb1kUy+Z2hInEHraAmivOtSlnaOPZ9mE4fMv/GMTepsmX/XjI88606ky55K
D3UXletByaTwe+dykOujJs3E0dYcWtJSJMy/CHMd0EG6tTBVrde8NYgnWKkixUqHTWsScuDR1iUB
AIf3nJ4BrZ2PleFijdoCkp36qiGHwFa8NHxMnInZ0s3CKEKmHe+KcZPzuqwmm44GvqGAX3I/VYAA
AAAAAAAMgAAAAQAAAI6AAAACgAAAA4AAAASAAAAFAAAAAV9Qam8AAAABYR1ShwAAAACdxfYxAAAA
ANz1Di4AAAABPUnxJAAAAADNxzlGr6vCJpIFz4XaG/fi/f9C9zgQ8ptKSQpfQ1NMJBGTDTxxYGGp
ch2xUA==
`
