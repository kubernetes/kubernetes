// This work is subject to the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
// license. Its contents can be found at:
// http://creativecommons.org/publicdomain/zero/1.0/

package bindata

import (
	"bytes"
	"fmt"
	"go/format"
	"io"
	"sort"
	"strings"
)

type assetTree struct {
	Asset    Asset
	Children map[string]*assetTree
}

func newAssetTree() *assetTree {
	tree := &assetTree{}
	tree.Children = make(map[string]*assetTree)
	return tree
}

func (node *assetTree) child(name string) *assetTree {
	rv, ok := node.Children[name]
	if !ok {
		rv = newAssetTree()
		// TODO: maintain this in sorted order
		node.Children[name] = rv
	}
	return rv
}

func (root *assetTree) Add(route []string, asset Asset) {
	for _, name := range route {
		root = root.child(name)
	}
	root.Asset = asset
}

func ident(buf *bytes.Buffer, n int) {
	for i := 0; i < n; i++ {
		buf.WriteByte('\t')
	}
}

func (root *assetTree) funcOrNil() string {
	if root.Asset.Func == "" {
		return "nil"
	} else {
		return root.Asset.Func
	}
}

func (root *assetTree) writeGoMap(buf *bytes.Buffer, nident int) {
	buf.Grow(35) // at least this size
	if nident == 0 {
		// at the top level we need to declare the map type
		buf.WriteString("&bintree")
	}
	fmt.Fprintf(buf, "{%s, map[string]*bintree{", root.funcOrNil())

	if len(root.Children) > 0 {
		buf.WriteByte('\n')

		// Sort to make output stable between invocations
		filenames := make([]string, len(root.Children))
		i := 0
		for filename := range root.Children {
			filenames[i] = filename
			i++
		}
		sort.Strings(filenames)

		for _, p := range filenames {
			ident(buf, nident+1)
			buf.WriteByte('"')
			buf.WriteString(p)
			buf.WriteString(`": `)
			root.Children[p].writeGoMap(buf, nident+1)
		}
		ident(buf, nident)
	}

	buf.WriteString("}}")
	if nident > 0 {
		buf.WriteByte(',')
	}
	buf.WriteByte('\n')
}

func (root *assetTree) WriteAsGoMap(w io.Writer) error {
	_, err := w.Write([]byte(`type bintree struct {
	Func     func() (*asset, error)
	Children map[string]*bintree
}

var _bintree = `))
	if err != nil {
		return err
	}
	buf := new(bytes.Buffer)
	root.writeGoMap(buf, 0)
	_, writeErr := w.Write(buf.Bytes())
	return writeErr
}

func writeTOCTree(w io.Writer, toc []Asset) error {
	_, err := w.Write([]byte(`// AssetDir returns the file names below a certain
// directory embedded in the file by go-bindata.
// For example if you run go-bindata on data/... and data contains the
// following hierarchy:
//     data/
//       foo.txt
//       img/
//         a.png
//         b.png
// then AssetDir("data") would return []string{"foo.txt", "img"},
// AssetDir("data/img") would return []string{"a.png", "b.png"},
// AssetDir("foo.txt") and AssetDir("notexist") would return an error, and
// AssetDir("") will return []string{"data"}.
func AssetDir(name string) ([]string, error) {
	node := _bintree
	if len(name) != 0 {
		canonicalName := strings.Replace(name, "\\", "/", -1)
		pathList := strings.Split(canonicalName, "/")
		for _, p := range pathList {
			node = node.Children[p]
			if node == nil {
				return nil, fmt.Errorf("Asset %s not found", name)
			}
		}
	}
	if node.Func != nil {
		return nil, fmt.Errorf("Asset %s not found", name)
	}
	rv := make([]string, 0, len(node.Children))
	for childName := range node.Children {
		rv = append(rv, childName)
	}
	return rv, nil
}

`))
	if err != nil {
		return err
	}
	tree := newAssetTree()
	for i := range toc {
		pathList := strings.Split(toc[i].Name, "/")
		tree.Add(pathList, toc[i])
	}
	return tree.WriteAsGoMap(w)
}

// writeTOC writes the table of contents file.
func writeTOC(buf *bytes.Buffer, toc []Asset) error {
	writeTOCHeader(buf)

	// Need an innerBuf so we can call gofmt and get map keys formatted at the
	// appropriate indentation.
	innerBuf := new(strings.Builder)
	if err := writeTOCMapHeader(innerBuf); err != nil {
		return err
	}

	for i := range toc {
		if err := writeTOCAsset(innerBuf, &toc[i]); err != nil {
			return err
		}
	}

	writeTOCFooter(innerBuf)
	fmted, err := format.Source([]byte(innerBuf.String()))
	if err != nil {
		return err
	}
	if _, err := buf.Write(fmted); err != nil {
		return err
	}
	return nil
}

// writeTOCHeader writes the table of contents file header.
func writeTOCHeader(buf *bytes.Buffer) {
	buf.WriteString(`// Asset loads and returns the asset for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func Asset(name string) ([]byte, error) {
	canonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[canonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("Asset %s can't read by error: %v", name, err)
		}
		return a.bytes, nil
	}
	return nil, fmt.Errorf("Asset %s not found", name)
}

// AssetString returns the asset contents as a string (instead of a []byte).
func AssetString(name string) (string, error) {
	data, err := Asset(name)
	return string(data), err
}

// MustAsset is like Asset but panics when Asset would return an error.
// It simplifies safe initialization of global variables.
func MustAsset(name string) []byte {
	a, err := Asset(name)
	if err != nil {
		panic("asset: Asset(" + name + "): " + err.Error())
	}

	return a
}

// MustAssetString is like AssetString but panics when Asset would return an
// error. It simplifies safe initialization of global variables.
func MustAssetString(name string) string {
	return string(MustAsset(name))
}

// AssetInfo loads and returns the asset info for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func AssetInfo(name string) (os.FileInfo, error) {
	canonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[canonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("AssetInfo %s can't read by error: %v", name, err)
		}
		return a.info, nil
	}
	return nil, fmt.Errorf("AssetInfo %s not found", name)
}

// AssetDigest returns the digest of the file with the given name. It returns an
// error if the asset could not be found or the digest could not be loaded.
func AssetDigest(name string) ([sha256.Size]byte, error) {
	canonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[canonicalName]; ok {
		a, err := f()
		if err != nil {
			return [sha256.Size]byte{}, fmt.Errorf("AssetDigest %s can't read by error: %v", name, err)
		}
		return a.digest, nil
	}
	return [sha256.Size]byte{}, fmt.Errorf("AssetDigest %s not found", name)
}

// Digests returns a map of all known files and their checksums.
func Digests() (map[string][sha256.Size]byte, error) {
	mp := make(map[string][sha256.Size]byte, len(_bindata))
	for name := range _bindata {
		a, err := _bindata[name]()
		if err != nil {
			return nil, err
		}
		mp[name] = a.digest
	}
	return mp, nil
}

// AssetNames returns the names of the assets.
func AssetNames() []string {
	names := make([]string, 0, len(_bindata))
	for name := range _bindata {
		names = append(names, name)
	}
	return names
}
`)
}

func writeTOCMapHeader(w io.StringWriter) error {
	_, err := w.WriteString(`
// _bindata is a table, holding each asset generator, mapped to its name.
var _bindata = map[string]func() (*asset, error){
`)
	return err
}

// writeTOCAsset writes a TOC entry for the given asset.
func writeTOCAsset(w io.Writer, asset *Asset) error {
	_, err := fmt.Fprintf(w, "\t%q: %s,\n", asset.Name, asset.Func)
	return err
}

// writeTOCFooter writes the table of contents file footer.
func writeTOCFooter(w io.StringWriter) {
	w.WriteString(`}

`)
}
