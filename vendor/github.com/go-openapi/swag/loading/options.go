// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package loading

import (
	"errors"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

type (
	// Option provides options for loading a file over HTTP or from a file.
	Option func(*options)

	httpOptions struct {
		httpTimeout       time.Duration
		basicAuthUsername string
		basicAuthPassword string
		customHeaders     map[string]string
		client            *http.Client
	}

	fileOptions struct {
		fs   fs.ReadFileFS
		root string // when non-empty, local reads are confined to this directory via os.Root
	}

	options struct {
		httpOptions
		fileOptions
	}
)

func (fo fileOptions) ReadFileFunc() func(string) ([]byte, error) {
	if fo.root != "" {
		root := fo.root

		return func(name string) ([]byte, error) {
			// os.Root only accepts paths relative to the root, but callers (and this package's
			// own file:// handling) routinely produce absolute paths. Rebase an absolute path
			// onto the root before handing it to os.Root.
			rel, err := rootRelative(root, name)
			if err != nil {
				return nil, errors.Join(err, ErrLoader)
			}

			r, err := os.OpenRoot(root)
			if err != nil {
				return nil, errors.Join(err, ErrLoader)
			}
			defer func() { _ = r.Close() }()

			return r.ReadFile(rel)
		}
	}

	if fo.fs == nil {
		return os.ReadFile
	}

	return fo.fs.ReadFile
}

// rootRelative expresses name as a path relative to root, so that it can be resolved by os.Root.
//
// A relative name is returned unchanged: os.Root confines it directly (including "../" traversal
// and symlink escapes, which it rejects at open time).
//
// An absolute name is rebased onto root. If it cannot be expressed relative to root — for
// example because it lives on a different volume on Windows — filepath.Rel returns an error,
// which is propagated so the read is rejected rather than silently escaping the root. An
// absolute path that lexically escapes root yields a "../" prefix here and is then rejected by
// os.Root.
func rootRelative(root, name string) (string, error) {
	osName := toOSPath(name)
	if !filepath.IsAbs(osName) {
		return name, nil
	}

	absRoot, err := filepath.Abs(toOSPath(root))
	if err != nil {
		return "", err
	}

	return filepath.Rel(absRoot, osName)
}

// toOSPath converts a slash-separated path to an OS-native path.
//
// On Windows it additionally normalizes the "/C:/dir" form — a leading separator before a drive
// letter — that file URIs and URL-style path normalization (as performed by
// github.com/go-openapi/spec) produce. Without this, filepath.IsAbs does not recognize such a
// path as absolute and os.Root rejects an otherwise in-root target. This mirrors the file://
// drive-letter handling in LoadStrategy, which the os.Root loader bypasses.
func toOSPath(p string) string {
	p = filepath.FromSlash(p)
	if runtime.GOOS == "windows" {
		p = stripLeadingDriveSlash(p)
	}

	return p
}

// stripLeadingDriveSlash removes a leading separator that precedes a Windows drive letter,
// turning "\C:\dir" (from a "/C:/dir" URL path) into "C:\dir". Any other path is returned
// unchanged. It is pure (no OS dependency) so that its logic can be tested on any platform.
func stripLeadingDriveSlash(p string) string {
	if len(p) >= 3 && (p[0] == '/' || p[0] == '\\') && p[2] == ':' && isASCIILetter(p[1]) {
		return p[1:]
	}

	return p
}

func isASCIILetter(b byte) bool {
	return ('a' <= b && b <= 'z') || ('A' <= b && b <= 'Z')
}

// WithTimeout sets a timeout for the remote file loader.
//
// The default timeout is 30s.
func WithTimeout(timeout time.Duration) Option {
	return func(o *options) {
		o.httpTimeout = timeout
	}
}

// WithBasicAuth sets a basic authentication scheme for the remote file loader.
func WithBasicAuth(username, password string) Option {
	return func(o *options) {
		o.basicAuthUsername = username
		o.basicAuthPassword = password
	}
}

// WithCustomHeaders sets custom headers for the remote file loader.
func WithCustomHeaders(headers map[string]string) Option {
	return func(o *options) {
		if o.customHeaders == nil {
			o.customHeaders = make(map[string]string, len(headers))
		}

		for header, value := range headers {
			o.customHeaders[header] = value
		}
	}
}

// WithHTTPClient overrides the default HTTP client used to fetch a remote file.
//
// By default, [http.DefaultClient] is used.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithFS sets a file system for the local file loader.
//
// If the provided file system is a [fs.ReadFileFS], the ReadFile function is used.
// Otherwise, ReadFile is wrapped using [fs.ReadFile].
//
// By default, the file system is the one provided by the os package.
//
// For example, this may be set to consume from an embedded file system, or a rooted FS.
//
// WithFS and [WithRoot] are mutually exclusive: the last one applied wins.
//
// Security note: a file system built from [os.DirFS] confines paths but does NOT protect
// against symlinks that escape the root. To load from a directory derived from untrusted
// input, prefer [WithRoot], which is symlink-escape resistant.
func WithFS(filesystem fs.FS) Option {
	return func(o *options) {
		o.root = "" // last-wins vs WithRoot
		if rfs, ok := filesystem.(fs.ReadFileFS); ok {
			o.fs = rfs

			return
		}
		o.fs = readFileFS{FS: filesystem}
	}
}

// WithRoot confines local file loading to dir.
//
// Every requested path is resolved within dir. A relative path is resolved against dir; an
// absolute path is rebased onto dir (so a caller that normalizes references to absolute paths,
// such as github.com/go-openapi/spec, still resolves correctly). Any path that would escape dir
// — through ".." traversal, an absolute path pointing outside dir, or a symlink pointing outside
// dir — is rejected. This is built on [os.Root] and is therefore resistant to the symlink
// escapes that a plain [os.DirFS] does not prevent.
//
// WithRoot is the recommended option when loading specs from a location derived from
// untrusted input. It applies to local loading only and has no effect on remote
// (http/https) loading. WithRoot and [WithFS] are mutually exclusive: the last one applied
// wins.
//
// Note: [os.Root] confines path resolution but does not, by itself, protect against
// traversal of mount/bind boundaries, /proc special files, or device files. Point WithRoot
// at a directory that holds only the documents you intend to expose.
func WithRoot(dir string) Option {
	return func(o *options) {
		o.root = dir
		o.fs = nil // last-wins vs WithFS
	}
}

type readFileFS struct {
	fs.FS
}

func (r readFileFS) ReadFile(name string) ([]byte, error) {
	return fs.ReadFile(r.FS, name)
}

func optionsWithDefaults(opts []Option) options {
	const defaultTimeout = 30 * time.Second

	o := options{
		// package level defaults
		httpOptions: httpOptions{
			httpTimeout: defaultTimeout,
			client:      http.DefaultClient,
		},
	}

	for _, apply := range opts {
		apply(&o)
	}

	return o
}
