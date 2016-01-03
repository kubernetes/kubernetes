package negroni

import (
	"net/http"
	"path"
	"strings"
)

// Static is a middleware handler that serves static files in the given directory/filesystem.
type Static struct {
	// Dir is the directory to serve static files from
	Dir http.FileSystem
	// Prefix is the optional prefix used to serve the static directory content
	Prefix string
	// IndexFile defines which file to serve as index if it exists.
	IndexFile string
}

// NewStatic returns a new instance of Static
func NewStatic(directory http.FileSystem) *Static {
	return &Static{
		Dir:       directory,
		Prefix:    "",
		IndexFile: "index.html",
	}
}

func (s *Static) ServeHTTP(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
	if r.Method != "GET" && r.Method != "HEAD" {
		next(rw, r)
		return
	}
	file := r.URL.Path
	// if we have a prefix, filter requests by stripping the prefix
	if s.Prefix != "" {
		if !strings.HasPrefix(file, s.Prefix) {
			next(rw, r)
			return
		}
		file = file[len(s.Prefix):]
		if file != "" && file[0] != '/' {
			next(rw, r)
			return
		}
	}
	f, err := s.Dir.Open(file)
	if err != nil {
		// discard the error?
		next(rw, r)
		return
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		next(rw, r)
		return
	}

	// try to serve index file
	if fi.IsDir() {
		// redirect if missing trailing slash
		if !strings.HasSuffix(r.URL.Path, "/") {
			http.Redirect(rw, r, r.URL.Path+"/", http.StatusFound)
			return
		}

		file = path.Join(file, s.IndexFile)
		f, err = s.Dir.Open(file)
		if err != nil {
			next(rw, r)
			return
		}
		defer f.Close()

		fi, err = f.Stat()
		if err != nil || fi.IsDir() {
			next(rw, r)
			return
		}
	}

	http.ServeContent(rw, r, file, fi.ModTime(), f)
}
