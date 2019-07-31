/*
Copyright 2017 The Kubernetes Authors.

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

package disk

import (
	"net/http"
	"os"
	"path/filepath"

	"github.com/gregjones/httpcache"
	"github.com/gregjones/httpcache/diskcache"
	"github.com/peterbourgon/diskv"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/uuid"
)

type cacheRoundTripper struct {
	rt *httpcache.Transport
}

// newCacheRoundTripper creates a roundtripper that reads the ETag on
// response headers and send the If-None-Match header on subsequent
// corresponding requests.
func newCacheRoundTripper(cacheDir string, rt http.RoundTripper) http.RoundTripper {
	perms := os.FileMode(0660)
	// if the directory exists or can be created...
	if err := os.MkdirAll(cacheDir, 0750); err == nil {
		// and we can create a tmp file to check default permissions...
		if f, err := os.Create(filepath.Join(cacheDir, "umask.tmp."+string(uuid.NewUUID()))); err == nil {
			// clean up the tmp file when we're done
			defer os.Remove(f.Name())
			// determine default file permission (honoring umask), dropping world-readable permissions
			if info, err := os.Stat(f.Name()); err == nil {
				perms = info.Mode().Perm() & os.FileMode(0660)
			}
		}
	}

	d := diskv.New(diskv.Options{
		PathPerm: os.FileMode(0750),
		FilePerm: perms,
		BasePath: cacheDir,
		TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
	})
	t := httpcache.NewTransport(diskcache.NewWithDiskv(d))
	t.Transport = rt

	return &cacheRoundTripper{rt: t}
}

func (rt *cacheRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return rt.rt.RoundTrip(req)
}

func (rt *cacheRoundTripper) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := rt.rt.Transport.(canceler); ok {
		cr.CancelRequest(req)
	} else {
		klog.Errorf("CancelRequest not implemented by %T", rt.rt.Transport)
	}
}

func (rt *cacheRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt.Transport }
