// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webdav

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/net/context"
)

func TestMemPS(t *testing.T) {
	ctx := context.Background()
	// calcProps calculates the getlastmodified and getetag DAV: property
	// values in pstats for resource name in file-system fs.
	calcProps := func(name string, fs FileSystem, ls LockSystem, pstats []Propstat) error {
		fi, err := fs.Stat(ctx, name)
		if err != nil {
			return err
		}
		for _, pst := range pstats {
			for i, p := range pst.Props {
				switch p.XMLName {
				case xml.Name{Space: "DAV:", Local: "getlastmodified"}:
					p.InnerXML = []byte(fi.ModTime().Format(http.TimeFormat))
					pst.Props[i] = p
				case xml.Name{Space: "DAV:", Local: "getetag"}:
					if fi.IsDir() {
						continue
					}
					etag, err := findETag(ctx, fs, ls, name, fi)
					if err != nil {
						return err
					}
					p.InnerXML = []byte(etag)
					pst.Props[i] = p
				}
			}
		}
		return nil
	}

	const (
		lockEntry = `` +
			`<D:lockentry xmlns:D="DAV:">` +
			`<D:lockscope><D:exclusive/></D:lockscope>` +
			`<D:locktype><D:write/></D:locktype>` +
			`</D:lockentry>`
		statForbiddenError = `<D:cannot-modify-protected-property xmlns:D="DAV:"/>`
	)

	type propOp struct {
		op            string
		name          string
		pnames        []xml.Name
		patches       []Proppatch
		wantPnames    []xml.Name
		wantPropstats []Propstat
	}

	testCases := []struct {
		desc        string
		noDeadProps bool
		buildfs     []string
		propOp      []propOp
	}{{
		desc:    "propname",
		buildfs: []string{"mkdir /dir", "touch /file"},
		propOp: []propOp{{
			op:   "propname",
			name: "/dir",
			wantPnames: []xml.Name{
				{Space: "DAV:", Local: "resourcetype"},
				{Space: "DAV:", Local: "displayname"},
				{Space: "DAV:", Local: "supportedlock"},
				{Space: "DAV:", Local: "getlastmodified"},
			},
		}, {
			op:   "propname",
			name: "/file",
			wantPnames: []xml.Name{
				{Space: "DAV:", Local: "resourcetype"},
				{Space: "DAV:", Local: "displayname"},
				{Space: "DAV:", Local: "getcontentlength"},
				{Space: "DAV:", Local: "getlastmodified"},
				{Space: "DAV:", Local: "getcontenttype"},
				{Space: "DAV:", Local: "getetag"},
				{Space: "DAV:", Local: "supportedlock"},
			},
		}},
	}, {
		desc:    "allprop dir and file",
		buildfs: []string{"mkdir /dir", "write /file foobarbaz"},
		propOp: []propOp{{
			op:   "allprop",
			name: "/dir",
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "resourcetype"},
					InnerXML: []byte(`<D:collection xmlns:D="DAV:"/>`),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "displayname"},
					InnerXML: []byte("dir"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getlastmodified"},
					InnerXML: nil, // Calculated during test.
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "supportedlock"},
					InnerXML: []byte(lockEntry),
				}},
			}},
		}, {
			op:   "allprop",
			name: "/file",
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "resourcetype"},
					InnerXML: []byte(""),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "displayname"},
					InnerXML: []byte("file"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getcontentlength"},
					InnerXML: []byte("9"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getlastmodified"},
					InnerXML: nil, // Calculated during test.
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getcontenttype"},
					InnerXML: []byte("text/plain; charset=utf-8"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getetag"},
					InnerXML: nil, // Calculated during test.
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "supportedlock"},
					InnerXML: []byte(lockEntry),
				}},
			}},
		}, {
			op:   "allprop",
			name: "/file",
			pnames: []xml.Name{
				{"DAV:", "resourcetype"},
				{"foo", "bar"},
			},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "resourcetype"},
					InnerXML: []byte(""),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "displayname"},
					InnerXML: []byte("file"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getcontentlength"},
					InnerXML: []byte("9"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getlastmodified"},
					InnerXML: nil, // Calculated during test.
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getcontenttype"},
					InnerXML: []byte("text/plain; charset=utf-8"),
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "getetag"},
					InnerXML: nil, // Calculated during test.
				}, {
					XMLName:  xml.Name{Space: "DAV:", Local: "supportedlock"},
					InnerXML: []byte(lockEntry),
				}}}, {
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}}},
			},
		}},
	}, {
		desc:    "propfind DAV:resourcetype",
		buildfs: []string{"mkdir /dir", "touch /file"},
		propOp: []propOp{{
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{"DAV:", "resourcetype"}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "resourcetype"},
					InnerXML: []byte(`<D:collection xmlns:D="DAV:"/>`),
				}},
			}},
		}, {
			op:     "propfind",
			name:   "/file",
			pnames: []xml.Name{{"DAV:", "resourcetype"}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "resourcetype"},
					InnerXML: []byte(""),
				}},
			}},
		}},
	}, {
		desc:    "propfind unsupported DAV properties",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{"DAV:", "getcontentlanguage"}},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "getcontentlanguage"},
				}},
			}},
		}, {
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{"DAV:", "creationdate"}},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "creationdate"},
				}},
			}},
		}},
	}, {
		desc:    "propfind getetag for files but not for directories",
		buildfs: []string{"mkdir /dir", "touch /file"},
		propOp: []propOp{{
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{"DAV:", "getetag"}},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "getetag"},
				}},
			}},
		}, {
			op:     "propfind",
			name:   "/file",
			pnames: []xml.Name{{"DAV:", "getetag"}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "getetag"},
					InnerXML: nil, // Calculated during test.
				}},
			}},
		}},
	}, {
		desc:        "proppatch property on no-dead-properties file system",
		buildfs:     []string{"mkdir /dir"},
		noDeadProps: true,
		propOp: []propOp{{
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusForbidden,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}, {
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "getetag"},
				}},
			}},
			wantPropstats: []Propstat{{
				Status:   http.StatusForbidden,
				XMLError: statForbiddenError,
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "getetag"},
				}},
			}},
		}},
	}, {
		desc:    "proppatch dead property",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}, {
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{Space: "foo", Local: "bar"}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}},
			}},
		}},
	}, {
		desc:    "proppatch dead property with failed dependency",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}},
			}, {
				Props: []Property{{
					XMLName:  xml.Name{Space: "DAV:", Local: "displayname"},
					InnerXML: []byte("xxx"),
				}},
			}},
			wantPropstats: []Propstat{{
				Status:   http.StatusForbidden,
				XMLError: statForbiddenError,
				Props: []Property{{
					XMLName: xml.Name{Space: "DAV:", Local: "displayname"},
				}},
			}, {
				Status: StatusFailedDependency,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}, {
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{Space: "foo", Local: "bar"}},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}},
	}, {
		desc:    "proppatch remove dead property",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}, {
					XMLName:  xml.Name{Space: "spam", Local: "ham"},
					InnerXML: []byte("eggs"),
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}, {
					XMLName: xml.Name{Space: "spam", Local: "ham"},
				}},
			}},
		}, {
			op:   "propfind",
			name: "/dir",
			pnames: []xml.Name{
				{Space: "foo", Local: "bar"},
				{Space: "spam", Local: "ham"},
			},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}, {
					XMLName:  xml.Name{Space: "spam", Local: "ham"},
					InnerXML: []byte("eggs"),
				}},
			}},
		}, {
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Remove: true,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}, {
			op:   "propfind",
			name: "/dir",
			pnames: []xml.Name{
				{Space: "foo", Local: "bar"},
				{Space: "spam", Local: "ham"},
			},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}, {
				Status: http.StatusOK,
				Props: []Property{{
					XMLName:  xml.Name{Space: "spam", Local: "ham"},
					InnerXML: []byte("eggs"),
				}},
			}},
		}},
	}, {
		desc:    "propname with dead property",
		buildfs: []string{"touch /file"},
		propOp: []propOp{{
			op:   "proppatch",
			name: "/file",
			patches: []Proppatch{{
				Props: []Property{{
					XMLName:  xml.Name{Space: "foo", Local: "bar"},
					InnerXML: []byte("baz"),
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}, {
			op:   "propname",
			name: "/file",
			wantPnames: []xml.Name{
				{Space: "DAV:", Local: "resourcetype"},
				{Space: "DAV:", Local: "displayname"},
				{Space: "DAV:", Local: "getcontentlength"},
				{Space: "DAV:", Local: "getlastmodified"},
				{Space: "DAV:", Local: "getcontenttype"},
				{Space: "DAV:", Local: "getetag"},
				{Space: "DAV:", Local: "supportedlock"},
				{Space: "foo", Local: "bar"},
			},
		}},
	}, {
		desc:    "proppatch remove unknown dead property",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:   "proppatch",
			name: "/dir",
			patches: []Proppatch{{
				Remove: true,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
			wantPropstats: []Propstat{{
				Status: http.StatusOK,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo", Local: "bar"},
				}},
			}},
		}},
	}, {
		desc:    "bad: propfind unknown property",
		buildfs: []string{"mkdir /dir"},
		propOp: []propOp{{
			op:     "propfind",
			name:   "/dir",
			pnames: []xml.Name{{"foo:", "bar"}},
			wantPropstats: []Propstat{{
				Status: http.StatusNotFound,
				Props: []Property{{
					XMLName: xml.Name{Space: "foo:", Local: "bar"},
				}},
			}},
		}},
	}}

	for _, tc := range testCases {
		fs, err := buildTestFS(tc.buildfs)
		if err != nil {
			t.Fatalf("%s: cannot create test filesystem: %v", tc.desc, err)
		}
		if tc.noDeadProps {
			fs = noDeadPropsFS{fs}
		}
		ls := NewMemLS()
		for _, op := range tc.propOp {
			desc := fmt.Sprintf("%s: %s %s", tc.desc, op.op, op.name)
			if err = calcProps(op.name, fs, ls, op.wantPropstats); err != nil {
				t.Fatalf("%s: calcProps: %v", desc, err)
			}

			// Call property system.
			var propstats []Propstat
			switch op.op {
			case "propname":
				pnames, err := propnames(ctx, fs, ls, op.name)
				if err != nil {
					t.Errorf("%s: got error %v, want nil", desc, err)
					continue
				}
				sort.Sort(byXMLName(pnames))
				sort.Sort(byXMLName(op.wantPnames))
				if !reflect.DeepEqual(pnames, op.wantPnames) {
					t.Errorf("%s: pnames\ngot  %q\nwant %q", desc, pnames, op.wantPnames)
				}
				continue
			case "allprop":
				propstats, err = allprop(ctx, fs, ls, op.name, op.pnames)
			case "propfind":
				propstats, err = props(ctx, fs, ls, op.name, op.pnames)
			case "proppatch":
				propstats, err = patch(ctx, fs, ls, op.name, op.patches)
			default:
				t.Fatalf("%s: %s not implemented", desc, op.op)
			}
			if err != nil {
				t.Errorf("%s: got error %v, want nil", desc, err)
				continue
			}
			// Compare return values from allprop, propfind or proppatch.
			for _, pst := range propstats {
				sort.Sort(byPropname(pst.Props))
			}
			for _, pst := range op.wantPropstats {
				sort.Sort(byPropname(pst.Props))
			}
			sort.Sort(byStatus(propstats))
			sort.Sort(byStatus(op.wantPropstats))
			if !reflect.DeepEqual(propstats, op.wantPropstats) {
				t.Errorf("%s: propstat\ngot  %q\nwant %q", desc, propstats, op.wantPropstats)
			}
		}
	}
}

func cmpXMLName(a, b xml.Name) bool {
	if a.Space != b.Space {
		return a.Space < b.Space
	}
	return a.Local < b.Local
}

type byXMLName []xml.Name

func (b byXMLName) Len() int           { return len(b) }
func (b byXMLName) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byXMLName) Less(i, j int) bool { return cmpXMLName(b[i], b[j]) }

type byPropname []Property

func (b byPropname) Len() int           { return len(b) }
func (b byPropname) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byPropname) Less(i, j int) bool { return cmpXMLName(b[i].XMLName, b[j].XMLName) }

type byStatus []Propstat

func (b byStatus) Len() int           { return len(b) }
func (b byStatus) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byStatus) Less(i, j int) bool { return b[i].Status < b[j].Status }

type noDeadPropsFS struct {
	FileSystem
}

func (fs noDeadPropsFS) OpenFile(ctx context.Context, name string, flag int, perm os.FileMode) (File, error) {
	f, err := fs.FileSystem.OpenFile(ctx, name, flag, perm)
	if err != nil {
		return nil, err
	}
	return noDeadPropsFile{f}, nil
}

// noDeadPropsFile wraps a File but strips any optional DeadPropsHolder methods
// provided by the underlying File implementation.
type noDeadPropsFile struct {
	f File
}

func (f noDeadPropsFile) Close() error                              { return f.f.Close() }
func (f noDeadPropsFile) Read(p []byte) (int, error)                { return f.f.Read(p) }
func (f noDeadPropsFile) Readdir(count int) ([]os.FileInfo, error)  { return f.f.Readdir(count) }
func (f noDeadPropsFile) Seek(off int64, whence int) (int64, error) { return f.f.Seek(off, whence) }
func (f noDeadPropsFile) Stat() (os.FileInfo, error)                { return f.f.Stat() }
func (f noDeadPropsFile) Write(p []byte) (int, error)               { return f.f.Write(p) }
