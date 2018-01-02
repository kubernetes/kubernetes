package memfs

import (
	"testing"

	"gopkg.in/src-d/go-billy.v3/test"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type MemorySuite struct {
	test.FilesystemSuite
	path string
}

var _ = Suite(&MemorySuite{})

func (s *MemorySuite) SetUpTest(c *C) {
	s.FilesystemSuite = test.NewFilesystemSuite(New())
}
