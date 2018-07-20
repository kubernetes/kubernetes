package graphtest

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/containerd/continuity/driver"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/stringid"
)

func randomContent(size int, seed int64) []byte {
	s := rand.NewSource(seed)
	content := make([]byte, size)

	for i := 0; i < len(content); i += 7 {
		val := s.Int63()
		for j := 0; i+j < len(content) && j < 7; j++ {
			content[i+j] = byte(val)
			val >>= 8
		}
	}

	return content
}

func addFiles(drv graphdriver.Driver, layer string, seed int64) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	if err := driver.WriteFile(root, root.Join(root.Path(), "file-a"), randomContent(64, seed), 0755); err != nil {
		return err
	}
	if err := root.MkdirAll(root.Join(root.Path(), "dir-b"), 0755); err != nil {
		return err
	}
	if err := driver.WriteFile(root, root.Join(root.Path(), "dir-b", "file-b"), randomContent(128, seed+1), 0755); err != nil {
		return err
	}

	return driver.WriteFile(root, root.Join(root.Path(), "file-c"), randomContent(128*128, seed+2), 0755)
}

func checkFile(drv graphdriver.Driver, layer, filename string, content []byte) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	fileContent, err := driver.ReadFile(root, root.Join(root.Path(), filename))
	if err != nil {
		return err
	}

	if !bytes.Equal(fileContent, content) {
		return fmt.Errorf("mismatched file content %v, expecting %v", fileContent, content)
	}

	return nil
}

func addFile(drv graphdriver.Driver, layer, filename string, content []byte) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	return driver.WriteFile(root, root.Join(root.Path(), filename), content, 0755)
}

func addDirectory(drv graphdriver.Driver, layer, dir string) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	return root.MkdirAll(root.Join(root.Path(), dir), 0755)
}

func removeAll(drv graphdriver.Driver, layer string, names ...string) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	for _, filename := range names {
		if err := root.RemoveAll(root.Join(root.Path(), filename)); err != nil {
			return err
		}
	}
	return nil
}

func checkFileRemoved(drv graphdriver.Driver, layer, filename string) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	if _, err := root.Stat(root.Join(root.Path(), filename)); err == nil {
		return fmt.Errorf("file still exists: %s", root.Join(root.Path(), filename))
	} else if !os.IsNotExist(err) {
		return err
	}

	return nil
}

func addManyFiles(drv graphdriver.Driver, layer string, count int, seed int64) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	for i := 0; i < count; i += 100 {
		dir := root.Join(root.Path(), fmt.Sprintf("directory-%d", i))
		if err := root.MkdirAll(dir, 0755); err != nil {
			return err
		}
		for j := 0; i+j < count && j < 100; j++ {
			file := root.Join(dir, fmt.Sprintf("file-%d", i+j))
			if err := driver.WriteFile(root, file, randomContent(64, seed+int64(i+j)), 0755); err != nil {
				return err
			}
		}
	}

	return nil
}

func changeManyFiles(drv graphdriver.Driver, layer string, count int, seed int64) ([]archive.Change, error) {
	root, err := drv.Get(layer, "")
	if err != nil {
		return nil, err
	}
	defer drv.Put(layer)

	changes := []archive.Change{}
	for i := 0; i < count; i += 100 {
		archiveRoot := fmt.Sprintf("/directory-%d", i)
		if err := root.MkdirAll(root.Join(root.Path(), archiveRoot), 0755); err != nil {
			return nil, err
		}
		for j := 0; i+j < count && j < 100; j++ {
			if j == 0 {
				changes = append(changes, archive.Change{
					Path: archiveRoot,
					Kind: archive.ChangeModify,
				})
			}
			var change archive.Change
			switch j % 3 {
			// Update file
			case 0:
				change.Path = root.Join(archiveRoot, fmt.Sprintf("file-%d", i+j))
				change.Kind = archive.ChangeModify
				if err := driver.WriteFile(root, root.Join(root.Path(), change.Path), randomContent(64, seed+int64(i+j)), 0755); err != nil {
					return nil, err
				}
			// Add file
			case 1:
				change.Path = root.Join(archiveRoot, fmt.Sprintf("file-%d-%d", seed, i+j))
				change.Kind = archive.ChangeAdd
				if err := driver.WriteFile(root, root.Join(root.Path(), change.Path), randomContent(64, seed+int64(i+j)), 0755); err != nil {
					return nil, err
				}
			// Remove file
			case 2:
				change.Path = root.Join(archiveRoot, fmt.Sprintf("file-%d", i+j))
				change.Kind = archive.ChangeDelete
				if err := root.Remove(root.Join(root.Path(), change.Path)); err != nil {
					return nil, err
				}
			}
			changes = append(changes, change)
		}
	}

	return changes, nil
}

func checkManyFiles(drv graphdriver.Driver, layer string, count int, seed int64) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	for i := 0; i < count; i += 100 {
		dir := root.Join(root.Path(), fmt.Sprintf("directory-%d", i))
		for j := 0; i+j < count && j < 100; j++ {
			file := root.Join(dir, fmt.Sprintf("file-%d", i+j))
			fileContent, err := driver.ReadFile(root, file)
			if err != nil {
				return err
			}

			content := randomContent(64, seed+int64(i+j))

			if !bytes.Equal(fileContent, content) {
				return fmt.Errorf("mismatched file content %v, expecting %v", fileContent, content)
			}
		}
	}

	return nil
}

type changeList []archive.Change

func (c changeList) Less(i, j int) bool {
	if c[i].Path == c[j].Path {
		return c[i].Kind < c[j].Kind
	}
	return c[i].Path < c[j].Path
}
func (c changeList) Len() int      { return len(c) }
func (c changeList) Swap(i, j int) { c[j], c[i] = c[i], c[j] }

func checkChanges(expected, actual []archive.Change) error {
	if len(expected) != len(actual) {
		return fmt.Errorf("unexpected number of changes, expected %d, got %d", len(expected), len(actual))
	}
	sort.Sort(changeList(expected))
	sort.Sort(changeList(actual))

	for i := range expected {
		if expected[i] != actual[i] {
			return fmt.Errorf("unexpected change, expecting %v, got %v", expected[i], actual[i])
		}
	}

	return nil
}

func addLayerFiles(drv graphdriver.Driver, layer, parent string, i int) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	if err := driver.WriteFile(root, root.Join(root.Path(), "top-id"), []byte(layer), 0755); err != nil {
		return err
	}
	layerDir := root.Join(root.Path(), fmt.Sprintf("layer-%d", i))
	if err := root.MkdirAll(layerDir, 0755); err != nil {
		return err
	}
	if err := driver.WriteFile(root, root.Join(layerDir, "layer-id"), []byte(layer), 0755); err != nil {
		return err
	}
	if err := driver.WriteFile(root, root.Join(layerDir, "parent-id"), []byte(parent), 0755); err != nil {
		return err
	}

	return nil
}

func addManyLayers(drv graphdriver.Driver, baseLayer string, count int) (string, error) {
	lastLayer := baseLayer
	for i := 1; i <= count; i++ {
		nextLayer := stringid.GenerateRandomID()
		if err := drv.Create(nextLayer, lastLayer, nil); err != nil {
			return "", err
		}
		if err := addLayerFiles(drv, nextLayer, lastLayer, i); err != nil {
			return "", err
		}

		lastLayer = nextLayer

	}
	return lastLayer, nil
}

func checkManyLayers(drv graphdriver.Driver, layer string, count int) error {
	root, err := drv.Get(layer, "")
	if err != nil {
		return err
	}
	defer drv.Put(layer)

	layerIDBytes, err := driver.ReadFile(root, root.Join(root.Path(), "top-id"))
	if err != nil {
		return err
	}

	if !bytes.Equal(layerIDBytes, []byte(layer)) {
		return fmt.Errorf("mismatched file content %v, expecting %v", layerIDBytes, []byte(layer))
	}

	for i := count; i > 0; i-- {
		layerDir := root.Join(root.Path(), fmt.Sprintf("layer-%d", i))

		thisLayerIDBytes, err := driver.ReadFile(root, root.Join(layerDir, "layer-id"))
		if err != nil {
			return err
		}
		if !bytes.Equal(thisLayerIDBytes, layerIDBytes) {
			return fmt.Errorf("mismatched file content %v, expecting %v", thisLayerIDBytes, layerIDBytes)
		}
		layerIDBytes, err = driver.ReadFile(root, root.Join(layerDir, "parent-id"))
		if err != nil {
			return err
		}
	}
	return nil
}

// readDir reads a directory just like driver.ReadDir()
// then hides specific files (currently "lost+found")
// so the tests don't "see" it
func readDir(r driver.Driver, dir string) ([]os.FileInfo, error) {
	a, err := driver.ReadDir(r, dir)
	if err != nil {
		return nil, err
	}

	b := a[:0]
	for _, x := range a {
		if x.Name() != "lost+found" { // ext4 always have this dir
			b = append(b, x)
		}
	}

	return b, nil
}
