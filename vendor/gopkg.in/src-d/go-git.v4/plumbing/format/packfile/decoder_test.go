package packfile_test

import (
	"io"

	"gopkg.in/src-d/go-billy.v3/memfs"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/idxfile"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	. "gopkg.in/check.v1"
)

type ReaderSuite struct {
	fixtures.Suite
}

var _ = Suite(&ReaderSuite{})

func (s *ReaderSuite) TestNewDecodeNonSeekable(c *C) {
	scanner := packfile.NewScanner(nil)
	d, err := packfile.NewDecoder(scanner, nil)

	c.Assert(d, IsNil)
	c.Assert(err, NotNil)
}

func (s *ReaderSuite) TestDecode(c *C) {
	fixtures.Basic().ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		scanner := packfile.NewScanner(f.Packfile())
		storage := memory.NewStorage()

		d, err := packfile.NewDecoder(scanner, storage)
		c.Assert(err, IsNil)
		defer d.Close()

		ch, err := d.Decode()
		c.Assert(err, IsNil)
		c.Assert(ch, Equals, f.PackfileHash)

		assertObjects(c, storage, expectedHashes)
	})
}

func (s *ReaderSuite) TestDecodeByTypeRefDelta(c *C) {
	f := fixtures.Basic().ByTag("ref-delta").One()

	storage := memory.NewStorage()
	scanner := packfile.NewScanner(f.Packfile())
	d, err := packfile.NewDecoderForType(scanner, storage, plumbing.CommitObject)
	c.Assert(err, IsNil)

	// Index required to decode by ref-delta.
	d.SetIndex(getIndexFromIdxFile(f.Idx()))

	defer d.Close()

	_, count, err := scanner.Header()
	c.Assert(err, IsNil)

	var i uint32
	for i = 0; i < count; i++ {
		obj, err := d.DecodeObject()
		c.Assert(err, IsNil)

		if obj != nil {
			c.Assert(obj.Type(), Equals, plumbing.CommitObject)
		}
	}
}

func (s *ReaderSuite) TestDecodeByTypeRefDeltaError(c *C) {
	fixtures.Basic().ByTag("ref-delta").Test(c, func(f *fixtures.Fixture) {
		storage := memory.NewStorage()
		scanner := packfile.NewScanner(f.Packfile())
		d, err := packfile.NewDecoderForType(scanner, storage, plumbing.CommitObject)
		c.Assert(err, IsNil)

		defer d.Close()

		_, count, err := scanner.Header()
		c.Assert(err, IsNil)

		isError := false
		var i uint32
		for i = 0; i < count; i++ {
			_, err := d.DecodeObject()
			if err != nil {
				isError = true
				break
			}
		}
		c.Assert(isError, Equals, true)
	})

}

func (s *ReaderSuite) TestDecodeByType(c *C) {
	ts := []plumbing.ObjectType{
		plumbing.CommitObject,
		plumbing.TagObject,
		plumbing.TreeObject,
		plumbing.BlobObject,
	}

	fixtures.Basic().ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		for _, t := range ts {
			storage := memory.NewStorage()
			scanner := packfile.NewScanner(f.Packfile())
			d, err := packfile.NewDecoderForType(scanner, storage, t)
			c.Assert(err, IsNil)

			// when the packfile is ref-delta based, the offsets are required
			if f.Is("ref-delta") {
				d.SetIndex(getIndexFromIdxFile(f.Idx()))
			}

			defer d.Close()

			_, count, err := scanner.Header()
			c.Assert(err, IsNil)

			var i uint32
			for i = 0; i < count; i++ {
				obj, err := d.DecodeObject()
				c.Assert(err, IsNil)

				if obj != nil {
					c.Assert(obj.Type(), Equals, t)
				}
			}
		}
	})
}
func (s *ReaderSuite) TestDecodeByTypeConstructor(c *C) {
	f := fixtures.Basic().ByTag("packfile").One()
	storage := memory.NewStorage()
	scanner := packfile.NewScanner(f.Packfile())

	_, err := packfile.NewDecoderForType(scanner, storage, plumbing.OFSDeltaObject)
	c.Assert(err, Equals, plumbing.ErrInvalidType)

	_, err = packfile.NewDecoderForType(scanner, storage, plumbing.REFDeltaObject)
	c.Assert(err, Equals, plumbing.ErrInvalidType)

	_, err = packfile.NewDecoderForType(scanner, storage, plumbing.InvalidObject)
	c.Assert(err, Equals, plumbing.ErrInvalidType)
}

func (s *ReaderSuite) TestDecodeMultipleTimes(c *C) {
	f := fixtures.Basic().ByTag("packfile").One()
	scanner := packfile.NewScanner(f.Packfile())
	storage := memory.NewStorage()

	d, err := packfile.NewDecoder(scanner, storage)
	c.Assert(err, IsNil)
	defer d.Close()

	ch, err := d.Decode()
	c.Assert(err, IsNil)
	c.Assert(ch, Equals, f.PackfileHash)

	ch, err = d.Decode()
	c.Assert(err, Equals, packfile.ErrAlreadyDecoded)
	c.Assert(ch, Equals, plumbing.ZeroHash)
}

func (s *ReaderSuite) TestDecodeInMemory(c *C) {
	fixtures.Basic().ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		scanner := packfile.NewScanner(f.Packfile())
		d, err := packfile.NewDecoder(scanner, nil)
		c.Assert(err, IsNil)

		ch, err := d.Decode()
		c.Assert(err, IsNil)
		c.Assert(ch, Equals, f.PackfileHash)
	})
}

type nonSeekableReader struct {
	r io.Reader
}

func (nsr nonSeekableReader) Read(b []byte) (int, error) {
	return nsr.r.Read(b)
}

func (s *ReaderSuite) TestDecodeNoSeekableWithTxStorer(c *C) {
	fixtures.Basic().ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		reader := nonSeekableReader{
			r: f.Packfile(),
		}

		scanner := packfile.NewScanner(reader)

		var storage storer.EncodedObjectStorer = memory.NewStorage()
		_, isTxStorer := storage.(storer.Transactioner)
		c.Assert(isTxStorer, Equals, true)

		d, err := packfile.NewDecoder(scanner, storage)
		c.Assert(err, IsNil)
		defer d.Close()

		ch, err := d.Decode()
		c.Assert(err, IsNil)
		c.Assert(ch, Equals, f.PackfileHash)

		assertObjects(c, storage, expectedHashes)
	})
}

func (s *ReaderSuite) TestDecodeNoSeekableWithoutTxStorer(c *C) {
	fixtures.Basic().ByTag("packfile").Test(c, func(f *fixtures.Fixture) {
		reader := nonSeekableReader{
			r: f.Packfile(),
		}

		scanner := packfile.NewScanner(reader)

		var storage storer.EncodedObjectStorer
		storage, _ = filesystem.NewStorage(memfs.New())
		_, isTxStorer := storage.(storer.Transactioner)
		c.Assert(isTxStorer, Equals, false)

		d, err := packfile.NewDecoder(scanner, storage)
		c.Assert(err, IsNil)
		defer d.Close()

		ch, err := d.Decode()
		c.Assert(err, IsNil)
		c.Assert(ch, Equals, f.PackfileHash)

		assertObjects(c, storage, expectedHashes)
	})
}

var expectedHashes = []string{
	"918c48b83bd081e863dbe1b80f8998f058cd8294",
	"af2d6a6954d532f8ffb47615169c8fdf9d383a1a",
	"1669dce138d9b841a518c64b10914d88f5e488ea",
	"a5b8b09e2f8fcb0bb99d3ccb0958157b40890d69",
	"b8e471f58bcbca63b07bda20e428190409c2db47",
	"35e85108805c84807bc66a02d91535e1e24b38b9",
	"b029517f6300c2da0f4b651b8642506cd6aaf45d",
	"32858aad3c383ed1ff0a0f9bdf231d54a00c9e88",
	"d3ff53e0564a9f87d8e84b6e28e5060e517008aa",
	"c192bd6a24ea1ab01d78686e417c8bdc7c3d197f",
	"d5c0f4ab811897cadf03aec358ae60d21f91c50d",
	"49c6bb89b17060d7b4deacb7b338fcc6ea2352a9",
	"cf4aa3b38974fb7d81f367c0830f7d78d65ab86b",
	"9dea2395f5403188298c1dabe8bdafe562c491e3",
	"586af567d0bb5e771e49bdd9434f5e0fb76d25fa",
	"9a48f23120e880dfbe41f7c9b7b708e9ee62a492",
	"5a877e6a906a2743ad6e45d99c1793642aaf8eda",
	"c8f1d8c61f9da76f4cb49fd86322b6e685dba956",
	"a8d315b2b1c615d43042c3a62402b8a54288cf5c",
	"a39771a7651f97faf5c72e08224d857fc35133db",
	"880cd14280f4b9b6ed3986d6671f907d7cc2a198",
	"fb72698cab7617ac416264415f13224dfd7a165e",
	"4d081c50e250fa32ea8b1313cf8bb7c2ad7627fd",
	"eba74343e2f15d62adedfd8c883ee0262b5c8021",
	"c2d30fa8ef288618f65f6eed6e168e0d514886f4",
	"8dcef98b1d52143e1e2dbc458ffe38f925786bf2",
	"aa9b383c260e1d05fbbf6b30a02914555e20c725",
	"6ecf0ef2c2dffb796033e5a02219af86ec6584e5",
	"dbd3641b371024f44d0e469a9c8f5457b0660de1",
	"e8d3ffab552895c19b9fcf7aa264d277cde33881",
	"7e59600739c96546163833214c36459e324bad0a",
}

func (s *ReaderSuite) TestDecodeCRCs(c *C) {
	f := fixtures.Basic().ByTag("ofs-delta").One()

	scanner := packfile.NewScanner(f.Packfile())
	storage := memory.NewStorage()

	d, err := packfile.NewDecoder(scanner, storage)
	c.Assert(err, IsNil)
	_, err = d.Decode()
	c.Assert(err, IsNil)

	var sum uint64
	idx := d.Index().ToIdxFile()
	for _, e := range idx.Entries {
		sum += uint64(e.CRC32)
	}

	c.Assert(int(sum), Equals, 78022211966)
}

func (s *ReaderSuite) TestReadObjectAt(c *C) {
	f := fixtures.Basic().One()
	scanner := packfile.NewScanner(f.Packfile())
	d, err := packfile.NewDecoder(scanner, nil)
	c.Assert(err, IsNil)

	// when the packfile is ref-delta based, the offsets are required
	if f.Is("ref-delta") {
		d.SetIndex(getIndexFromIdxFile(f.Idx()))
	}

	// the objects at reference 186, is a delta, so should be recall,
	// without being read before.
	obj, err := d.DecodeObjectAt(186)
	c.Assert(err, IsNil)
	c.Assert(obj.Hash().String(), Equals, "6ecf0ef2c2dffb796033e5a02219af86ec6584e5")
}

func (s *ReaderSuite) TestIndex(c *C) {
	f := fixtures.Basic().One()
	scanner := packfile.NewScanner(f.Packfile())
	d, err := packfile.NewDecoder(scanner, nil)
	c.Assert(err, IsNil)

	c.Assert(d.Index().ToIdxFile().Entries, HasLen, 0)

	_, err = d.Decode()
	c.Assert(err, IsNil)

	c.Assert(len(d.Index().ToIdxFile().Entries), Equals, 31)
}

func (s *ReaderSuite) TestSetIndex(c *C) {
	f := fixtures.Basic().One()
	scanner := packfile.NewScanner(f.Packfile())
	d, err := packfile.NewDecoder(scanner, nil)
	c.Assert(err, IsNil)

	idx := packfile.NewIndex(1)
	h := plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5")
	idx.Add(h, uint64(42), 0)
	d.SetIndex(idx)

	idxf := d.Index().ToIdxFile()
	c.Assert(idxf.Entries, HasLen, 1)
	c.Assert(idxf.Entries[0].Offset, Equals, uint64(42))
}

func assertObjects(c *C, s storer.EncodedObjectStorer, expects []string) {

	i, err := s.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)

	var count int
	err = i.ForEach(func(plumbing.EncodedObject) error { count++; return nil })
	c.Assert(err, IsNil)
	c.Assert(count, Equals, len(expects))

	for _, exp := range expects {
		obt, err := s.EncodedObject(plumbing.AnyObject, plumbing.NewHash(exp))
		c.Assert(err, IsNil)
		c.Assert(obt.Hash().String(), Equals, exp)
	}
}

func getIndexFromIdxFile(r io.Reader) *packfile.Index {
	idxf := idxfile.NewIdxfile()
	d := idxfile.NewDecoder(r)
	if err := d.Decode(idxf); err != nil {
		panic(err)
	}

	return packfile.NewIndexFromIdxFile(idxf)
}
