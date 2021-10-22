/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package simulator

import (
	"context"
	"net/http"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

func TestParseDatastorePath(t *testing.T) {
	tests := []struct {
		dsPath string
		dsFile string
		fail   bool
	}{
		{"", "", true},
		{"x", "", true},
		{"[", "", true},
		{"[nope", "", true},
		{"[test]", "", false},
		{"[test] foo", "foo", false},
		{"[test] foo/foo.vmx", "foo/foo.vmx", false},
		{"[test]foo bar/foo bar.vmx", "foo bar/foo bar.vmx", false},
	}

	for _, test := range tests {
		p, err := parseDatastorePath(test.dsPath)
		if test.fail {
			if err == nil {
				t.Errorf("expected error for: %s", test.dsPath)
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error '%#v' for: %s", err, test.dsPath)
			} else {
				if test.dsFile != p.Path {
					t.Errorf("dsFile=%s", p.Path)
				}
				if p.Datastore != "test" {
					t.Errorf("ds=%s", p.Datastore)
				}
			}
		}
	}
}

func TestRefreshDatastore(t *testing.T) {
	tests := []struct {
		dir  string
		fail bool
	}{
		{".", false},
		{"-", true},
	}

	for _, test := range tests {
		ds := &Datastore{}
		ds.Info = &types.LocalDatastoreInfo{
			DatastoreInfo: types.DatastoreInfo{
				Url: test.dir,
			},
		}

		res := ds.RefreshDatastore(nil)
		err := res.Fault()

		if test.fail {
			if err == nil {
				t.Error("expected error")
			}
		} else {
			if err != nil {
				t.Error(err)
			}
		}
	}
}

func TestDatastoreHTTP(t *testing.T) {
	ctx := context.Background()
	src := "datastore_test.go"
	dst := "tmp.go"

	for i, model := range []*Model{ESX(), VPX()} {
		defer model.Remove()
		err := model.Create()
		if err != nil {
			t.Fatal(err)
		}

		s := model.Service.NewServer()
		defer s.Close()

		if i == 0 {
			// Enable use of SessionManagerGenericServiceTicket.HostName in govmomi, disabled by default.
			opts := s.URL.Query()
			opts.Set("GOVMOMI_USE_SERVICE_TICKET_HOSTNAME", "true")
			s.URL.RawQuery = opts.Encode()
		}

		c, err := govmomi.NewClient(ctx, s.URL, true)
		if err != nil {
			t.Fatal(err)
		}

		finder := find.NewFinder(c.Client, false)

		dc, err := finder.DefaultDatacenter(ctx)
		if err != nil {
			t.Fatal(err)
		}

		finder.SetDatacenter(dc)

		ds, err := finder.DefaultDatastore(ctx)
		if err != nil {
			t.Fatal(err)
		}

		if i == 0 {
			// Cover the service ticket path
			// TODO: govmomi requires HostSystem.Config.VirtualNicManagerInfo
			// ctx = ds.HostContext(ctx, object.NewHostSystem(c.Client, esx.HostSystem.Reference()))
		}

		dsPath := ds.Path

		if !c.IsVC() {
			dc = nil // test using the default
		}

		fm := object.NewFileManager(c.Client)
		browser, err := ds.Browser(ctx)
		if err != nil {
			t.Fatal(err)
		}

		download := func(name string, fail bool) {
			st, serr := ds.Stat(ctx, name)

			_, _, err = ds.Download(ctx, name, nil)
			if fail {
				if err == nil {
					t.Fatal("expected Download error")
				}
				if serr == nil {
					t.Fatal("expected Stat error")
				}
			} else {
				if err != nil {
					t.Errorf("Download error: %s", err)
				}
				if serr != nil {
					t.Errorf("Stat error: %s", serr)
				}

				p := st.GetFileInfo().Path
				if p != name {
					t.Errorf("path=%s", p)
				}
			}
		}

		upload := func(name string, fail bool, method string) {
			f, err := os.Open(src)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			p := soap.DefaultUpload
			p.Method = method

			err = ds.Upload(ctx, f, name, &p)
			if fail {
				if err == nil {
					t.Fatalf("%s %s: expected error", method, name)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
			}
		}

		rm := func(name string, fail bool) {
			task, err := fm.DeleteDatastoreFile(ctx, dsPath(name), dc)
			if err != nil {
				t.Fatal(err)
			}

			err = task.Wait(ctx)
			if fail {
				if err == nil {
					t.Fatalf("rm %s: expected error", name)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
			}
		}

		mv := func(src string, dst string, fail bool, force bool) {
			task, err := fm.MoveDatastoreFile(ctx, dsPath(src), dc, dsPath(dst), dc, force)
			if err != nil {
				t.Fatal(err)
			}

			err = task.Wait(ctx)
			if fail {
				if err == nil {
					t.Fatalf("mv %s %s: expected error", src, dst)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
			}
		}

		mkdir := func(name string, fail bool, p bool) {
			err := fm.MakeDirectory(ctx, dsPath(name), dc, p)
			if fail {
				if err == nil {
					t.Fatalf("mkdir %s: expected error", name)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
			}
		}

		stat := func(name string, fail bool) {
			_, err := ds.Stat(ctx, name)
			if fail {
				if err == nil {
					t.Fatalf("stat %s: expected error", name)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
			}
		}

		ls := func(name string, fail bool) []types.BaseFileInfo {
			spec := types.HostDatastoreBrowserSearchSpec{
				MatchPattern: []string{"*"},
			}

			task, err := browser.SearchDatastore(ctx, dsPath(name), &spec)
			if err != nil {
				t.Fatal(err)
			}

			info, err := task.WaitForResult(ctx, nil)
			if err != nil {
				if fail {
					if err == nil {
						t.Fatalf("ls %s: expected error", name)
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
				}
				return nil
			}
			if info.Entity.Type != "Datastore" {
				t.Fatal(info.Entity.Type)
			}

			return info.Result.(types.HostDatastoreBrowserSearchResults).File
		}

		lsr := func(name string, fail bool, query ...types.BaseFileQuery) []types.HostDatastoreBrowserSearchResults {
			spec := types.HostDatastoreBrowserSearchSpec{
				MatchPattern: []string{"*"},
				Query:        query,
			}

			task, err := browser.SearchDatastoreSubFolders(ctx, dsPath(name), &spec)
			if err != nil {
				t.Fatal(err)
			}

			info, err := task.WaitForResult(ctx, nil)
			if err != nil {
				if fail {
					if err == nil {
						t.Fatalf("find %s: expected error", name)
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
				}
				return nil
			}
			if info.Entity.Type != "Datastore" {
				t.Fatal(info.Entity.Type)
			}

			return info.Result.(types.ArrayOfHostDatastoreBrowserSearchResults).HostDatastoreBrowserSearchResults
		}

		// GET file does not exist = fail
		download(dst, true)
		stat(dst, true)
		ls(dst, true)
		lsr(dst, true)

		// delete file does not exist = fail
		rm(dst, true)

		// PUT file = ok
		upload(dst, false, "PUT")
		stat(dst, false)
		ls("", false)
		lsr("", false)

		// GET file exists = ok
		download(dst, false)

		// POST file exists = fail
		upload(dst, true, "POST")

		// delete existing file = ok
		rm(dst, false)
		stat(dst, true)

		// GET file does not exist = fail
		download(dst, true)

		// POST file does not exist = ok
		upload(dst, false, "POST")

		// PATCH method not supported = fail
		upload(dst+".patch", true, "PATCH")

		// PUT path is directory = fail
		upload("", true, "PUT")

		// POST parent does not exist = ok
		upload("foobar/"+dst, false, "POST")

		// PUT parent does not exist = ok
		upload("barfoo/"+dst, false, "PUT")

		// mkdir parent does not exist = fail
		mkdir("foo/bar", true, false)

		// mkdir -p parent does not exist = ok
		mkdir("foo/bar", false, true)

		// mkdir = ok
		mkdir("foo/bar/baz", false, false)

		target := path.Join("foo", dst)

		// mv dst not exist = ok
		mv(dst, target, false, false)

		// POST file does not exist = ok
		upload(dst, false, "POST")

		// mv dst exists = fail
		mv(dst, target, true, false)

		// mv dst exists, force=true = ok
		mv(dst, target, false, true)

		// mv src does not exist = fail
		mv(dst, target, true, true)

		// ls -R = ok
		res := lsr("foo", false)

		count := func(s string) int {
			n := 0
			for _, dir := range res {
				for _, f := range dir.File {
					if strings.HasSuffix(f.GetFileInfo().Path, s) {
						n++
					}
				}
			}
			return n
		}

		n := len(res) + count("")

		if n != 6 {
			t.Errorf("ls -R foo==%d", n)
		}

		// test FileQuery
		res = lsr("", false)
		all := count(".vmdk") // foo-flat.vmdk + foo.vmdk

		res = lsr("", false, new(types.VmDiskFileQuery))
		allq := count(".vmdk") // foo.vmdk only
		if allq*2 != all {
			t.Errorf("ls -R *.vmdk: %d vs %d", all, allq)
		}

		res = lsr("", false, new(types.VmLogFileQuery), new(types.VmConfigFileQuery))
		all = count("")
		if all != model.Count().Machine*2 {
			t.Errorf("ls -R vmware.log+.vmx: %d", all)
		}

		invalid := []string{
			"",       //InvalidDatastorePath
			"[nope]", // InvalidDatastore
		}

		// test FileType details
		mkdir("exts", false, false)
		stat("exts", false)
		exts := []string{"img", "iso", "log", "nvram", "vmdk", "vmx"}
		for _, ext := range exts {
			name := dst + "." + ext
			upload(name, false, "POST")
			stat(name, false)
		}

		for _, p := range invalid {
			dsPath = func(name string) string {
				return p
			}
			mv(target, dst, true, false)
			mkdir("sorry", true, false)
			rm(target, true)
			ls(target, true)
		}

		// cover the dst failure path
		for _, p := range invalid {
			dsPath = func(name string) string {
				if name == dst {
					return p
				}
				return ds.Path(name)
			}
			mv(target, dst, true, false)
		}

		dsPath = func(name string) string {
			return ds.Path("enoent")
		}
		ls(target, true)

		// cover the case where datacenter or datastore lookup fails
		for _, q := range []string{"dcName=nope", "dsName=nope"} {
			u := *s.URL
			u.RawQuery = q
			u.Path = path.Join(folderPrefix, dst)

			r, err := http.Get(u.String())
			if err != nil {
				t.Fatal(err)
			}

			if r.StatusCode == http.StatusOK {
				t.Error("expected failure")
			}
		}
	}
}
