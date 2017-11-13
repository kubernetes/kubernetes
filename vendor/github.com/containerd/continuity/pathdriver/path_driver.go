package pathdriver

import (
	"path/filepath"
)

// PathDriver provides all of the path manipulation functions in a common
// interface. The context should call these and never use the `filepath`
// package or any other package to manipulate paths.
type PathDriver interface {
	Join(paths ...string) string
	IsAbs(path string) bool
	Rel(base, target string) (string, error)
	Base(path string) string
	Dir(path string) string
	Clean(path string) string
	Split(path string) (dir, file string)
	Separator() byte
	Abs(path string) (string, error)
	Walk(string, filepath.WalkFunc) error
	FromSlash(path string) string
	ToSlash(path string) string
	Match(pattern, name string) (matched bool, err error)
}

// pathDriver is a simple default implementation calls the filepath package.
type pathDriver struct{}

// LocalPathDriver is the exported pathDriver struct for convenience.
var LocalPathDriver PathDriver = &pathDriver{}

func (*pathDriver) Join(paths ...string) string {
	return filepath.Join(paths...)
}

func (*pathDriver) IsAbs(path string) bool {
	return filepath.IsAbs(path)
}

func (*pathDriver) Rel(base, target string) (string, error) {
	return filepath.Rel(base, target)
}

func (*pathDriver) Base(path string) string {
	return filepath.Base(path)
}

func (*pathDriver) Dir(path string) string {
	return filepath.Dir(path)
}

func (*pathDriver) Clean(path string) string {
	return filepath.Clean(path)
}

func (*pathDriver) Split(path string) (dir, file string) {
	return filepath.Split(path)
}

func (*pathDriver) Separator() byte {
	return filepath.Separator
}

func (*pathDriver) Abs(path string) (string, error) {
	return filepath.Abs(path)
}

// Note that filepath.Walk calls os.Stat, so if the context wants to
// to call Driver.Stat() for Walk, they need to create a new struct that
// overrides this method.
func (*pathDriver) Walk(root string, walkFn filepath.WalkFunc) error {
	return filepath.Walk(root, walkFn)
}

func (*pathDriver) FromSlash(path string) string {
	return filepath.FromSlash(path)
}

func (*pathDriver) ToSlash(path string) string {
	return filepath.ToSlash(path)
}

func (*pathDriver) Match(pattern, name string) (bool, error) {
	return filepath.Match(pattern, name)
}
