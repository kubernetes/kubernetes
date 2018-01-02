package object

import (
	"bytes"
	"io"
	"strings"
	"time"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
)

type SuiteCommit struct {
	BaseObjectsSuite
	Commit *Commit
}

var _ = Suite(&SuiteCommit{})

func (s *SuiteCommit) SetUpSuite(c *C) {
	s.BaseObjectsSuite.SetUpSuite(c)

	hash := plumbing.NewHash("1669dce138d9b841a518c64b10914d88f5e488ea")

	s.Commit = s.commit(c, hash)
}

func (s *SuiteCommit) TestDecodeNonCommit(c *C) {
	hash := plumbing.NewHash("9a48f23120e880dfbe41f7c9b7b708e9ee62a492")
	blob, err := s.Storer.EncodedObject(plumbing.AnyObject, hash)
	c.Assert(err, IsNil)

	commit := &Commit{}
	err = commit.Decode(blob)
	c.Assert(err, Equals, ErrUnsupportedObject)
}

func (s *SuiteCommit) TestType(c *C) {
	c.Assert(s.Commit.Type(), Equals, plumbing.CommitObject)
}

func (s *SuiteCommit) TestTree(c *C) {
	tree, err := s.Commit.Tree()
	c.Assert(err, IsNil)
	c.Assert(tree.ID().String(), Equals, "eba74343e2f15d62adedfd8c883ee0262b5c8021")
}

func (s *SuiteCommit) TestParents(c *C) {
	expected := []string{
		"35e85108805c84807bc66a02d91535e1e24b38b9",
		"a5b8b09e2f8fcb0bb99d3ccb0958157b40890d69",
	}

	var output []string
	i := s.Commit.Parents()
	err := i.ForEach(func(commit *Commit) error {
		output = append(output, commit.ID().String())
		return nil
	})

	c.Assert(err, IsNil)
	c.Assert(output, DeepEquals, expected)

	i.Close()
}

func (s *SuiteCommit) TestPatch(c *C) {
	from := s.commit(c, plumbing.NewHash("918c48b83bd081e863dbe1b80f8998f058cd8294"))
	to := s.commit(c, plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))

	patch, err := from.Patch(to)
	c.Assert(err, IsNil)

	buf := bytes.NewBuffer(nil)
	err = patch.Encode(buf)
	c.Assert(err, IsNil)

	c.Assert(buf.String(), Equals, `diff --git a/vendor/foo.go b/vendor/foo.go
new file mode 100644
index 0000000000000000000000000000000000000000..9dea2395f5403188298c1dabe8bdafe562c491e3
--- /dev/null
+++ b/vendor/foo.go
@@ -0,0 +1,7 @@
+package main
+
+import "fmt"
+
+func main() {
+	fmt.Println("Hello, playground")
+}
`)
	c.Assert(buf.String(), Equals, patch.String())

	from = s.commit(c, plumbing.NewHash("b8e471f58bcbca63b07bda20e428190409c2db47"))
	to = s.commit(c, plumbing.NewHash("35e85108805c84807bc66a02d91535e1e24b38b9"))

	patch, err = from.Patch(to)
	c.Assert(err, IsNil)

	buf.Reset()
	err = patch.Encode(buf)
	c.Assert(err, IsNil)

	c.Assert(buf.String(), Equals, `diff --git a/CHANGELOG b/CHANGELOG
deleted file mode 100644
index d3ff53e0564a9f87d8e84b6e28e5060e517008aa..0000000000000000000000000000000000000000
--- a/CHANGELOG
+++ /dev/null
@@ -1 +0,0 @@
-Initial changelog
diff --git a/binary.jpg b/binary.jpg
new file mode 100644
index 0000000000000000000000000000000000000000..d5c0f4ab811897cadf03aec358ae60d21f91c50d
Binary files /dev/null and b/binary.jpg differ
`)

	c.Assert(buf.String(), Equals, patch.String())
}

func (s *SuiteCommit) TestCommitEncodeDecodeIdempotent(c *C) {
	ts, err := time.Parse(time.RFC3339, "2006-01-02T15:04:05-07:00")
	c.Assert(err, IsNil)
	commits := []*Commit{
		{
			Author:       Signature{Name: "Foo", Email: "foo@example.local", When: ts},
			Committer:    Signature{Name: "Bar", Email: "bar@example.local", When: ts},
			Message:      "Message\n\nFoo\nBar\nWith trailing blank lines\n\n",
			TreeHash:     plumbing.NewHash("f000000000000000000000000000000000000001"),
			ParentHashes: []plumbing.Hash{plumbing.NewHash("f000000000000000000000000000000000000002")},
		},
		{
			Author:    Signature{Name: "Foo", Email: "foo@example.local", When: ts},
			Committer: Signature{Name: "Bar", Email: "bar@example.local", When: ts},
			Message:   "Message\n\nFoo\nBar\nWith no trailing blank lines",
			TreeHash:  plumbing.NewHash("0000000000000000000000000000000000000003"),
			ParentHashes: []plumbing.Hash{
				plumbing.NewHash("f000000000000000000000000000000000000004"),
				plumbing.NewHash("f000000000000000000000000000000000000005"),
				plumbing.NewHash("f000000000000000000000000000000000000006"),
				plumbing.NewHash("f000000000000000000000000000000000000007"),
			},
		},
	}
	for _, commit := range commits {
		obj := &plumbing.MemoryObject{}
		err = commit.Encode(obj)
		c.Assert(err, IsNil)
		newCommit := &Commit{}
		err = newCommit.Decode(obj)
		c.Assert(err, IsNil)
		commit.Hash = obj.Hash()
		c.Assert(newCommit, DeepEquals, commit)
	}
}

func (s *SuiteCommit) TestFile(c *C) {
	file, err := s.Commit.File("CHANGELOG")
	c.Assert(err, IsNil)
	c.Assert(file.Name, Equals, "CHANGELOG")
}

func (s *SuiteCommit) TestNumParents(c *C) {
	c.Assert(s.Commit.NumParents(), Equals, 2)
}

func (s *SuiteCommit) TestString(c *C) {
	c.Assert(s.Commit.String(), Equals, ""+
		"commit 1669dce138d9b841a518c64b10914d88f5e488ea\n"+
		"Author: Máximo Cuadros Ortiz <mcuadros@gmail.com>\n"+
		"Date:   Tue Mar 31 13:48:14 2015 +0200\n"+
		"\n"+
		"    Merge branch 'master' of github.com:tyba/git-fixture\n"+
		"\n",
	)
}

func (s *SuiteCommit) TestStringMultiLine(c *C) {
	hash := plumbing.NewHash("e7d896db87294e33ca3202e536d4d9bb16023db3")

	f := fixtures.ByURL("https://github.com/src-d/go-git.git").One()
	sto, err := filesystem.NewStorage(f.DotGit())

	o, err := sto.EncodedObject(plumbing.CommitObject, hash)
	c.Assert(err, IsNil)
	commit, err := DecodeCommit(sto, o)
	c.Assert(err, IsNil)

	c.Assert(commit.String(), Equals, ""+
		"commit e7d896db87294e33ca3202e536d4d9bb16023db3\n"+
		"Author: Alberto Cortés <alberto@sourced.tech>\n"+
		"Date:   Wed Jan 27 11:13:49 2016 +0100\n"+
		"\n"+
		"    fix zlib invalid header error\n"+
		"\n"+
		"    The return value of reads to the packfile were being ignored, so zlib\n"+
		"    was getting invalid data on it read buffers.\n"+
		"\n",
	)
}

func (s *SuiteCommit) TestCommitIterNext(c *C) {
	i := s.Commit.Parents()

	commit, err := i.Next()
	c.Assert(err, IsNil)
	c.Assert(commit.ID().String(), Equals, "35e85108805c84807bc66a02d91535e1e24b38b9")

	commit, err = i.Next()
	c.Assert(err, IsNil)
	c.Assert(commit.ID().String(), Equals, "a5b8b09e2f8fcb0bb99d3ccb0958157b40890d69")

	commit, err = i.Next()
	c.Assert(err, Equals, io.EOF)
	c.Assert(commit, IsNil)
}

func (s *SuiteCommit) TestLongCommitMessageSerialization(c *C) {
	encoded := &plumbing.MemoryObject{}
	decoded := &Commit{}
	commit := *s.Commit

	longMessage := "my message: message\n\n" + strings.Repeat("test", 4096) + "\nOK"
	commit.Message = longMessage

	err := commit.Encode(encoded)
	c.Assert(err, IsNil)

	err = decoded.Decode(encoded)
	c.Assert(err, IsNil)
	c.Assert(decoded.Message, Equals, longMessage)
}
