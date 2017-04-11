package inmemory

import (
	"fmt"
	"io"
	"path"
	"sort"
	"strings"
	"time"
)

var (
	errExists    = fmt.Errorf("exists")
	errNotExists = fmt.Errorf("notexists")
	errIsNotDir  = fmt.Errorf("notdir")
	errIsDir     = fmt.Errorf("isdir")
)

type node interface {
	name() string
	path() string
	isdir() bool
	modtime() time.Time
}

// dir is the central type for the memory-based  storagedriver. All operations
// are dispatched from a root dir.
type dir struct {
	common

	// TODO(stevvooe): Use sorted slice + search.
	children map[string]node
}

var _ node = &dir{}

func (d *dir) isdir() bool {
	return true
}

// add places the node n into dir d.
func (d *dir) add(n node) {
	if d.children == nil {
		d.children = make(map[string]node)
	}

	d.children[n.name()] = n
	d.mod = time.Now()
}

// find searches for the node, given path q in dir. If the node is found, it
// will be returned. If the node is not found, the closet existing parent. If
// the node is found, the returned (node).path() will match q.
func (d *dir) find(q string) node {
	q = strings.Trim(q, "/")
	i := strings.Index(q, "/")

	if q == "" {
		return d
	}

	if i == 0 {
		panic("shouldn't happen, no root paths")
	}

	var component string
	if i < 0 {
		// No more path components
		component = q
	} else {
		component = q[:i]
	}

	child, ok := d.children[component]
	if !ok {
		// Node was not found. Return p and the current node.
		return d
	}

	if child.isdir() {
		// traverse down!
		q = q[i+1:]
		return child.(*dir).find(q)
	}

	return child
}

func (d *dir) list(p string) ([]string, error) {
	n := d.find(p)

	if n.path() != p {
		return nil, errNotExists
	}

	if !n.isdir() {
		return nil, errIsNotDir
	}

	var children []string
	for _, child := range n.(*dir).children {
		children = append(children, child.path())
	}

	sort.Strings(children)
	return children, nil
}

// mkfile or return the existing one. returns an error if it exists and is a
// directory. Essentially, this is open or create.
func (d *dir) mkfile(p string) (*file, error) {
	n := d.find(p)
	if n.path() == p {
		if n.isdir() {
			return nil, errIsDir
		}

		return n.(*file), nil
	}

	dirpath, filename := path.Split(p)
	// Make any non-existent directories
	n, err := d.mkdirs(dirpath)
	if err != nil {
		return nil, err
	}

	dd := n.(*dir)
	n = &file{
		common: common{
			p:   path.Join(dd.path(), filename),
			mod: time.Now(),
		},
	}

	dd.add(n)
	return n.(*file), nil
}

// mkdirs creates any missing directory entries in p and returns the result.
func (d *dir) mkdirs(p string) (*dir, error) {
	p = normalize(p)

	n := d.find(p)

	if !n.isdir() {
		// Found something there
		return nil, errIsNotDir
	}

	if n.path() == p {
		return n.(*dir), nil
	}

	dd := n.(*dir)

	relative := strings.Trim(strings.TrimPrefix(p, n.path()), "/")

	if relative == "" {
		return dd, nil
	}

	components := strings.Split(relative, "/")
	for _, component := range components {
		d, err := dd.mkdir(component)

		if err != nil {
			// This should actually never happen, since there are no children.
			return nil, err
		}
		dd = d
	}

	return dd, nil
}

// mkdir creates a child directory under d with the given name.
func (d *dir) mkdir(name string) (*dir, error) {
	if name == "" {
		return nil, fmt.Errorf("invalid dirname")
	}

	_, ok := d.children[name]
	if ok {
		return nil, errExists
	}

	child := &dir{
		common: common{
			p:   path.Join(d.path(), name),
			mod: time.Now(),
		},
	}
	d.add(child)
	d.mod = time.Now()

	return child, nil
}

func (d *dir) move(src, dst string) error {
	dstDirname, _ := path.Split(dst)

	dp, err := d.mkdirs(dstDirname)
	if err != nil {
		return err
	}

	srcDirname, srcFilename := path.Split(src)
	sp := d.find(srcDirname)

	if normalize(srcDirname) != normalize(sp.path()) {
		return errNotExists
	}

	spd, ok := sp.(*dir)
	if !ok {
		return errIsNotDir // paranoid.
	}

	s, ok := spd.children[srcFilename]
	if !ok {
		return errNotExists
	}

	delete(spd.children, srcFilename)

	switch n := s.(type) {
	case *dir:
		n.p = dst
	case *file:
		n.p = dst
	}

	dp.add(s)

	return nil
}

func (d *dir) delete(p string) error {
	dirname, filename := path.Split(p)
	parent := d.find(dirname)

	if normalize(dirname) != normalize(parent.path()) {
		return errNotExists
	}

	if _, ok := parent.(*dir).children[filename]; !ok {
		return errNotExists
	}

	delete(parent.(*dir).children, filename)
	return nil
}

// dump outputs a primitive directory structure to stdout.
func (d *dir) dump(indent string) {
	fmt.Println(indent, d.name()+"/")

	for _, child := range d.children {
		if child.isdir() {
			child.(*dir).dump(indent + "\t")
		} else {
			fmt.Println(indent, child.name())
		}

	}
}

func (d *dir) String() string {
	return fmt.Sprintf("&dir{path: %v, children: %v}", d.p, d.children)
}

// file stores actual data in the fs tree. It acts like an open, seekable file
// where operations are conducted through ReadAt and WriteAt. Use it with
// SectionReader for the best effect.
type file struct {
	common
	data []byte
}

var _ node = &file{}

func (f *file) isdir() bool {
	return false
}

func (f *file) truncate() {
	f.data = f.data[:0]
}

func (f *file) sectionReader(offset int64) io.Reader {
	return io.NewSectionReader(f, offset, int64(len(f.data))-offset)
}

func (f *file) ReadAt(p []byte, offset int64) (n int, err error) {
	return copy(p, f.data[offset:]), nil
}

func (f *file) WriteAt(p []byte, offset int64) (n int, err error) {
	off := int(offset)
	if cap(f.data) < off+len(p) {
		data := make([]byte, len(f.data), off+len(p))
		copy(data, f.data)
		f.data = data
	}

	f.mod = time.Now()
	f.data = f.data[:off+len(p)]

	return copy(f.data[off:off+len(p)], p), nil
}

func (f *file) String() string {
	return fmt.Sprintf("&file{path: %q}", f.p)
}

// common provides shared fields and methods for node implementations.
type common struct {
	p   string
	mod time.Time
}

func (c *common) name() string {
	_, name := path.Split(c.p)
	return name
}

func (c *common) path() string {
	return c.p
}

func (c *common) modtime() time.Time {
	return c.mod
}

func normalize(p string) string {
	return "/" + strings.Trim(p, "/")
}
