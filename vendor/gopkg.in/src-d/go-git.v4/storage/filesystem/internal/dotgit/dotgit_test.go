package dotgit

import (
	"bufio"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-billy.v3/osfs"
)

func Test(t *testing.T) { TestingT(t) }

type SuiteDotGit struct {
	fixtures.Suite
}

var _ = Suite(&SuiteDotGit{})

func (s *SuiteDotGit) TestInitialize(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)

	err = dir.Initialize()
	c.Assert(err, IsNil)

	_, err = fs.Stat(fs.Join("objects", "info"))
	c.Assert(err, IsNil)

	_, err = fs.Stat(fs.Join("objects", "pack"))
	c.Assert(err, IsNil)

	_, err = fs.Stat(fs.Join("refs", "heads"))
	c.Assert(err, IsNil)

	_, err = fs.Stat(fs.Join("refs", "tags"))
	c.Assert(err, IsNil)
}

func (s *SuiteDotGit) TestSetRefs(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)

	err = dir.SetRef(plumbing.NewReferenceFromStrings(
		"refs/heads/foo",
		"e8d3ffab552895c19b9fcf7aa264d277cde33881",
	))

	c.Assert(err, IsNil)

	err = dir.SetRef(plumbing.NewReferenceFromStrings(
		"refs/heads/symbolic",
		"ref: refs/heads/foo",
	))

	c.Assert(err, IsNil)

	err = dir.SetRef(plumbing.NewReferenceFromStrings(
		"bar",
		"e8d3ffab552895c19b9fcf7aa264d277cde33881",
	))
	c.Assert(err, IsNil)

	refs, err := dir.Refs()
	c.Assert(err, IsNil)
	c.Assert(refs, HasLen, 2)

	ref := findReference(refs, "refs/heads/foo")
	c.Assert(ref, NotNil)
	c.Assert(ref.Hash().String(), Equals, "e8d3ffab552895c19b9fcf7aa264d277cde33881")

	ref = findReference(refs, "refs/heads/symbolic")
	c.Assert(ref, NotNil)
	c.Assert(ref.Target().String(), Equals, "refs/heads/foo")

	ref = findReference(refs, "bar")
	c.Assert(ref, IsNil)

	ref, err = dir.Ref("refs/heads/foo")
	c.Assert(err, IsNil)
	c.Assert(ref, NotNil)
	c.Assert(ref.Hash().String(), Equals, "e8d3ffab552895c19b9fcf7aa264d277cde33881")

	ref, err = dir.Ref("refs/heads/symbolic")
	c.Assert(err, IsNil)
	c.Assert(ref, NotNil)
	c.Assert(ref.Target().String(), Equals, "refs/heads/foo")

	ref, err = dir.Ref("bar")
	c.Assert(err, IsNil)
	c.Assert(ref, NotNil)
	c.Assert(ref.Hash().String(), Equals, "e8d3ffab552895c19b9fcf7aa264d277cde33881")

}

func (s *SuiteDotGit) TestRefsFromPackedRefs(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	refs, err := dir.Refs()
	c.Assert(err, IsNil)

	ref := findReference(refs, "refs/remotes/origin/branch")
	c.Assert(ref, NotNil)
	c.Assert(ref.Hash().String(), Equals, "e8d3ffab552895c19b9fcf7aa264d277cde33881")

}

func (s *SuiteDotGit) TestRefsFromReferenceFile(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	refs, err := dir.Refs()
	c.Assert(err, IsNil)

	ref := findReference(refs, "refs/remotes/origin/HEAD")
	c.Assert(ref, NotNil)
	c.Assert(ref.Type(), Equals, plumbing.SymbolicReference)
	c.Assert(string(ref.Target()), Equals, "refs/remotes/origin/master")

}

func BenchmarkRefMultipleTimes(b *testing.B) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	refname := plumbing.ReferenceName("refs/remotes/origin/branch")

	dir := New(fs)
	_, err := dir.Ref(refname)
	if err != nil {
		b.Fatalf("unexpected error: %s", err)
	}

	for i := 0; i < b.N; i++ {
		_, err := dir.Ref(refname)
		if err != nil {
			b.Fatalf("unexpected error: %s", err)
		}
	}
}

func (s *SuiteDotGit) TestRemoveRefFromReferenceFile(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	name := plumbing.ReferenceName("refs/remotes/origin/HEAD")
	err := dir.RemoveRef(name)
	c.Assert(err, IsNil)

	refs, err := dir.Refs()
	c.Assert(err, IsNil)

	ref := findReference(refs, string(name))
	c.Assert(ref, IsNil)
}

func (s *SuiteDotGit) TestRemoveRefFromPackedRefs(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	name := plumbing.ReferenceName("refs/remotes/origin/master")
	err := dir.RemoveRef(name)
	c.Assert(err, IsNil)

	b, err := ioutil.ReadFile(filepath.Join(fs.Root(), packedRefsPath))
	c.Assert(err, IsNil)

	c.Assert(string(b), Equals, ""+
		"# pack-refs with: peeled fully-peeled \n"+
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 refs/heads/master\n"+
		"e8d3ffab552895c19b9fcf7aa264d277cde33881 refs/remotes/origin/branch\n")
}

func (s *SuiteDotGit) TestRemoveRefNonExistent(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	packedRefs := filepath.Join(fs.Root(), packedRefsPath)
	before, err := ioutil.ReadFile(packedRefs)
	c.Assert(err, IsNil)

	name := plumbing.ReferenceName("refs/heads/nonexistent")
	err = dir.RemoveRef(name)
	c.Assert(err, IsNil)

	after, err := ioutil.ReadFile(packedRefs)
	c.Assert(err, IsNil)

	c.Assert(string(before), Equals, string(after))
}

func (s *SuiteDotGit) TestRemoveRefInvalidPackedRefs(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	packedRefs := filepath.Join(fs.Root(), packedRefsPath)
	brokenContent := "BROKEN STUFF REALLY BROKEN"

	err := ioutil.WriteFile(packedRefs, []byte(brokenContent), os.FileMode(0755))
	c.Assert(err, IsNil)

	name := plumbing.ReferenceName("refs/heads/nonexistent")
	err = dir.RemoveRef(name)
	c.Assert(err, NotNil)

	after, err := ioutil.ReadFile(filepath.Join(fs.Root(), packedRefsPath))
	c.Assert(err, IsNil)

	c.Assert(brokenContent, Equals, string(after))
}

func (s *SuiteDotGit) TestRemoveRefInvalidPackedRefs2(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	packedRefs := filepath.Join(fs.Root(), packedRefsPath)
	brokenContent := strings.Repeat("a", bufio.MaxScanTokenSize*2)

	err := ioutil.WriteFile(packedRefs, []byte(brokenContent), os.FileMode(0755))
	c.Assert(err, IsNil)

	name := plumbing.ReferenceName("refs/heads/nonexistent")
	err = dir.RemoveRef(name)
	c.Assert(err, NotNil)

	after, err := ioutil.ReadFile(filepath.Join(fs.Root(), packedRefsPath))
	c.Assert(err, IsNil)

	c.Assert(brokenContent, Equals, string(after))
}

func (s *SuiteDotGit) TestRefsFromHEADFile(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	refs, err := dir.Refs()
	c.Assert(err, IsNil)

	ref := findReference(refs, "HEAD")
	c.Assert(ref, NotNil)
	c.Assert(ref.Type(), Equals, plumbing.SymbolicReference)
	c.Assert(string(ref.Target()), Equals, "refs/heads/master")
}

func (s *SuiteDotGit) TestConfig(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	file, err := dir.Config()
	c.Assert(err, IsNil)
	c.Assert(filepath.Base(file.Name()), Equals, "config")
}

func (s *SuiteDotGit) TestConfigWriteAndConfig(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)

	f, err := dir.ConfigWriter()
	c.Assert(err, IsNil)

	_, err = f.Write([]byte("foo"))
	c.Assert(err, IsNil)

	f, err = dir.Config()
	c.Assert(err, IsNil)

	cnt, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)

	c.Assert(string(cnt), Equals, "foo")
}

func (s *SuiteDotGit) TestIndex(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	idx, err := dir.Index()
	c.Assert(err, IsNil)
	c.Assert(idx, NotNil)
}

func (s *SuiteDotGit) TestIndexWriteAndIndex(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)

	f, err := dir.IndexWriter()
	c.Assert(err, IsNil)

	_, err = f.Write([]byte("foo"))
	c.Assert(err, IsNil)

	f, err = dir.Index()
	c.Assert(err, IsNil)

	cnt, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)

	c.Assert(string(cnt), Equals, "foo")
}

func (s *SuiteDotGit) TestShallow(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	file, err := dir.Shallow()
	c.Assert(err, IsNil)
	c.Assert(file, IsNil)
}

func (s *SuiteDotGit) TestShallowWriteAndShallow(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)

	f, err := dir.ShallowWriter()
	c.Assert(err, IsNil)

	_, err = f.Write([]byte("foo"))
	c.Assert(err, IsNil)

	f, err = dir.Shallow()
	c.Assert(err, IsNil)

	cnt, err := ioutil.ReadAll(f)
	c.Assert(err, IsNil)

	c.Assert(string(cnt), Equals, "foo")
}

func findReference(refs []*plumbing.Reference, name string) *plumbing.Reference {
	n := plumbing.ReferenceName(name)
	for _, ref := range refs {
		if ref.Name() == n {
			return ref
		}
	}

	return nil
}

func (s *SuiteDotGit) TestObjectsPack(c *C) {
	f := fixtures.Basic().ByTag(".git").One()
	fs := f.DotGit()
	dir := New(fs)

	hashes, err := dir.ObjectPacks()
	c.Assert(err, IsNil)
	c.Assert(hashes, HasLen, 1)
	c.Assert(hashes[0], Equals, f.PackfileHash)
}

func (s *SuiteDotGit) TestObjectPack(c *C) {
	f := fixtures.Basic().ByTag(".git").One()
	fs := f.DotGit()
	dir := New(fs)

	pack, err := dir.ObjectPack(f.PackfileHash)
	c.Assert(err, IsNil)
	c.Assert(filepath.Ext(pack.Name()), Equals, ".pack")
}

func (s *SuiteDotGit) TestObjectPackIdx(c *C) {
	f := fixtures.Basic().ByTag(".git").One()
	fs := f.DotGit()
	dir := New(fs)

	idx, err := dir.ObjectPackIdx(f.PackfileHash)
	c.Assert(err, IsNil)
	c.Assert(filepath.Ext(idx.Name()), Equals, ".idx")
	c.Assert(idx.Close(), IsNil)
}

func (s *SuiteDotGit) TestObjectPackNotFound(c *C) {
	fs := fixtures.Basic().ByTag(".git").One().DotGit()
	dir := New(fs)

	pack, err := dir.ObjectPack(plumbing.ZeroHash)
	c.Assert(err, Equals, ErrPackfileNotFound)
	c.Assert(pack, IsNil)

	idx, err := dir.ObjectPackIdx(plumbing.ZeroHash)
	c.Assert(err, Equals, ErrPackfileNotFound)
	c.Assert(idx, IsNil)
}

func (s *SuiteDotGit) TestNewObject(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)
	w, err := dir.NewObject()
	c.Assert(err, IsNil)

	err = w.WriteHeader(plumbing.BlobObject, 14)
	n, err := w.Write([]byte("this is a test"))
	c.Assert(err, IsNil)
	c.Assert(n, Equals, 14)

	c.Assert(w.Hash().String(), Equals, "a8a940627d132695a9769df883f85992f0ff4a43")

	err = w.Close()
	c.Assert(err, IsNil)

	i, err := fs.Stat("objects/a8/a940627d132695a9769df883f85992f0ff4a43")
	c.Assert(err, IsNil)
	c.Assert(i.Size(), Equals, int64(34))
}

func (s *SuiteDotGit) TestObjects(c *C) {
	fs := fixtures.ByTag(".git").ByTag("unpacked").One().DotGit()
	dir := New(fs)

	hashes, err := dir.Objects()
	c.Assert(err, IsNil)
	c.Assert(hashes, HasLen, 187)
	c.Assert(hashes[0].String(), Equals, "0097821d427a3c3385898eb13b50dcbc8702b8a3")
	c.Assert(hashes[1].String(), Equals, "01d5fa556c33743006de7e76e67a2dfcd994ca04")
	c.Assert(hashes[2].String(), Equals, "03db8e1fbe133a480f2867aac478fd866686d69e")
}

func (s *SuiteDotGit) TestObjectsNoFolder(c *C) {
	tmp, err := ioutil.TempDir("", "dot-git")
	c.Assert(err, IsNil)
	defer os.RemoveAll(tmp)

	fs := osfs.New(tmp)
	dir := New(fs)
	hash, err := dir.Objects()
	c.Assert(err, IsNil)
	c.Assert(hash, HasLen, 0)
}

func (s *SuiteDotGit) TestObject(c *C) {
	fs := fixtures.ByTag(".git").ByTag("unpacked").One().DotGit()
	dir := New(fs)

	hash := plumbing.NewHash("03db8e1fbe133a480f2867aac478fd866686d69e")
	file, err := dir.Object(hash)
	c.Assert(err, IsNil)
	c.Assert(strings.HasSuffix(
		file.Name(), fs.Join("objects", "03", "db8e1fbe133a480f2867aac478fd866686d69e")),
		Equals, true,
	)
}

func (s *SuiteDotGit) TestObjectNotFound(c *C) {
	fs := fixtures.ByTag(".git").ByTag("unpacked").One().DotGit()
	dir := New(fs)

	hash := plumbing.NewHash("not-found-object")
	file, err := dir.Object(hash)
	c.Assert(err, NotNil)
	c.Assert(file, IsNil)
}

func (s *SuiteDotGit) TestSubmodules(c *C) {
	fs := fixtures.ByTag("submodule").One().DotGit()
	dir := New(fs)

	m, err := dir.Module("basic")
	c.Assert(err, IsNil)
	c.Assert(strings.HasSuffix(m.Root(), m.Join(".git", "modules", "basic")), Equals, true)
}
