package ruleguard

import (
	"path/filepath"

	"github.com/quasilyte/go-ruleguard/internal/golist"
)

func findBundleFiles(pkgPath string) ([]string, error) { // nolint
	pkg, err := golist.JSON(pkgPath)
	if err != nil {
		return nil, err
	}
	files := make([]string, 0, len(pkg.GoFiles))
	for _, f := range pkg.GoFiles {
		files = append(files, filepath.Join(pkg.Dir, f))
	}
	return files, nil
}
