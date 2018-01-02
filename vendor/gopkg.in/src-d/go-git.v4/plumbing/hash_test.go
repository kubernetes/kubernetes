package plumbing

import (
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type HashSuite struct{}

var _ = Suite(&HashSuite{})

func (s *HashSuite) TestComputeHash(c *C) {
	hash := ComputeHash(BlobObject, []byte(""))
	c.Assert(hash.String(), Equals, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

	hash = ComputeHash(BlobObject, []byte("Hello, World!\n"))
	c.Assert(hash.String(), Equals, "8ab686eafeb1f44702738c8b0f24f2567c36da6d")
}

func (s *HashSuite) TestNewHash(c *C) {
	hash := ComputeHash(BlobObject, []byte("Hello, World!\n"))

	c.Assert(hash, Equals, NewHash(hash.String()))
}

func (s *HashSuite) TestIsZero(c *C) {
	hash := NewHash("foo")
	c.Assert(hash.IsZero(), Equals, true)

	hash = NewHash("8ab686eafeb1f44702738c8b0f24f2567c36da6d")
	c.Assert(hash.IsZero(), Equals, false)
}

func (s *HashSuite) TestNewHasher(c *C) {
	content := "hasher test sample"
	hasher := NewHasher(BlobObject, int64(len(content)))
	hasher.Write([]byte(content))
	c.Assert(hasher.Sum().String(), Equals, "dc42c3cc80028d0ec61f0a6b24cadd1c195c4dfc")
}

func (s *HashSuite) TestHashesSort(c *C) {
	i := []Hash{
		NewHash("2222222222222222222222222222222222222222"),
		NewHash("1111111111111111111111111111111111111111"),
	}

	HashesSort(i)

	c.Assert(i[0], Equals, NewHash("1111111111111111111111111111111111111111"))
	c.Assert(i[1], Equals, NewHash("2222222222222222222222222222222222222222"))
}
