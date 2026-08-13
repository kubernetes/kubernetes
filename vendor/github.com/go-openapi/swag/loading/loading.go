// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package loading

import (
	"context"
	"embed"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

// LoadFromFileOrHTTP loads the bytes from a file or a remote http server based on the path passed in.
//
// Security: by default a local path is read with no confinement, so a caller-controlled path
// (including a "file://" URI or an absolute path) may read any file the process can access.
// When the path may derive from untrusted input, confine local loading with [WithRoot].
func LoadFromFileOrHTTP(pth string, opts ...Option) ([]byte, error) {
	o := optionsWithDefaults(opts)
	return LoadStrategy(pth, o.ReadFileFunc(), loadHTTPBytes(opts...), opts...)(pth)
}

// LoadStrategy returns a loader function for a given path or URI.
//
// The load strategy returns the remote load for any path starting with `http`.
// So this works for any URI with a scheme `http` or `https`.
//
// The fallback strategy is to call the local loader.
//
// The local loader takes a local file system path (absolute or relative) as argument,
// or alternatively a `file://...` URI, **without host** (see also below for windows).
//
// There are a few liberalities, initially intended to be tolerant regarding the URI syntax,
// especially on windows.
//
// Before the local loader is called, the given path is transformed:
//   - percent-encoded characters are unescaped
//   - simple paths (e.g. `./folder/file`) are passed as-is
//   - on windows, occurrences of `/` are replaced by `\`, so providing a relative path such a `folder/file` works too.
//
// For paths provided as URIs with the "file" scheme, please note that:
//   - `file://` is simply stripped.
//     This means that the host part of the URI is not parsed at all.
//     For example, `file:///folder/file" becomes "/folder/file`,
//     but `file://localhost/folder/file` becomes `localhost/folder/file` on unix systems.
//     Similarly, `file://./folder/file` yields `./folder/file`.
//   - on windows, `file://...` can take a host so as to specify an UNC share location.
//
// Reminder about windows-specifics:
// - `file://host/folder/file` becomes an UNC path like `\\host\folder\file` (no port specification is supported)
// - `file:///c:/folder/file` becomes `C:\folder\file`
// - `file://c:/folder/file` is tolerated (without leading `/`) and becomes `c:\folder\file`
func LoadStrategy(pth string, local, remote func(string) ([]byte, error), opts ...Option) func(string) ([]byte, error) {
	if hasHTTPScheme(pth) {
		return remote
	}
	o := optionsWithDefaults(opts)
	_, isEmbedFS := o.fs.(embed.FS)
	// any loader backed by an fs.FS or an os.Root consumes forward-slash paths on every
	// platform, so it must not go through the windows-native file:// preprocessing below.
	isFSBacked := o.fs != nil || o.root != ""

	return func(p string) ([]byte, error) {
		upth, err := url.PathUnescape(p)
		if err != nil {
			return nil, err
		}

		cpth, hasPrefix := strings.CutPrefix(upth, "file://")
		if !hasPrefix || isFSBacked || runtime.GOOS != "windows" {
			// crude processing: trim the file:// prefix. This leaves full URIs with a host with a (mostly) unexpected result
			// regular file path provided: just normalize slashes
			if isEmbedFS {
				// embed.FS always uses "/" as separator, even on windows, and rejects leading "./" or "/".
				return local(strings.TrimLeft(filepath.ToSlash(cpth), "./")) // remove invalid leading characters for embed FS
			}

			if isFSBacked {
				// other fs.FS (e.g. os.DirFS) and os.Root loaders also use "/" on every platform.
				// Path confinement is enforced by the loader, not here: the os.Root loader rebases
				// absolute in-root paths and rejects escaping paths ("..", out-of-root absolute,
				// escaping symlinks); an fs.FS loader rejects what its file system does not allow.
				return local(filepath.ToSlash(cpth))
			}

			return local(filepath.FromSlash(cpth))
		}

		// windows-only pre-processing of file://... URIs, excluding embed.FS

		// support for canonical file URIs on windows.
		u, err := url.Parse(filepath.ToSlash(upth))
		if err != nil {
			return nil, err
		}

		if u.Host != "" {
			// assume UNC name (volume share)
			// NOTE: UNC port not yet supported

			// when the "host" segment is a drive letter:
			// file://C:/folder/... => C:\folder
			upth = path.Clean(strings.Join([]string{u.Host, u.Path}, `/`))
			if !strings.HasSuffix(u.Host, ":") && u.Host[0] != '.' {
				// tolerance: if we have a leading dot, this can't be a host
				// file://host/share/folder\... ==> \\host\share\path\folder
				upth = "//" + upth
			}
		} else {
			// no host, let's figure out if this is a drive letter
			upth = strings.TrimPrefix(upth, `file://`)
			first, _, _ := strings.Cut(strings.TrimPrefix(u.Path, "/"), "/")
			if strings.HasSuffix(first, ":") {
				// drive letter in the first segment:
				// file:///c:/folder/... ==> strip the leading slash
				upth = strings.TrimPrefix(upth, `/`)
			}
		}

		return local(filepath.FromSlash(upth))
	}
}

// hasHTTPScheme reports whether pth is an absolute URL with an http or https scheme,
// selecting the remote loader. The comparison is case-insensitive, as URL schemes are.
//
// Requiring the "://" separator (rather than a bare "http" prefix) avoids misrouting a
// local file whose name merely starts with "http" (e.g. "httpbin.json") to the remote loader.
func hasHTTPScheme(pth string) bool {
	for _, scheme := range [...]string{"http://", "https://"} {
		if len(pth) >= len(scheme) && strings.EqualFold(pth[:len(scheme)], scheme) {
			return true
		}
	}

	return false
}

func loadHTTPBytes(opts ...Option) func(path string) ([]byte, error) {
	o := optionsWithDefaults(opts)

	return func(path string) ([]byte, error) {
		client := o.client
		timeoutCtx := context.Background()
		var cancel func()

		if o.httpTimeout > 0 {
			timeoutCtx, cancel = context.WithTimeout(timeoutCtx, o.httpTimeout)
			defer cancel()
		}

		req, err := http.NewRequestWithContext(timeoutCtx, http.MethodGet, path, nil)
		if err != nil {
			return nil, err
		}

		if o.basicAuthUsername != "" && o.basicAuthPassword != "" {
			req.SetBasicAuth(o.basicAuthUsername, o.basicAuthPassword)
		}

		for key, val := range o.customHeaders {
			req.Header.Set(key, val)
		}

		resp, err := client.Do(req)
		defer func() {
			if resp != nil {
				if e := resp.Body.Close(); e != nil {
					log.Println(e)
				}
			}
		}()
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("could not access document at %q [%s]: %w", path, resp.Status, ErrLoader)
		}

		return io.ReadAll(resp.Body)
	}
}
