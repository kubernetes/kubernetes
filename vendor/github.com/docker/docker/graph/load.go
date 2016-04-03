// +build linux windows

package graph

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
)

// Loads a set of images into the repository. This is the complementary of ImageExport.
// The input stream is an uncompressed tar ball containing images and metadata.
func (s *TagStore) Load(inTar io.ReadCloser, outStream io.Writer) error {
	tmpImageDir, err := ioutil.TempDir("", "docker-import-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpImageDir)

	var (
		repoDir = filepath.Join(tmpImageDir, "repo")
	)

	if err := os.Mkdir(repoDir, os.ModeDir); err != nil {
		return err
	}
	images := s.graph.Map()
	excludes := make([]string, len(images))
	i := 0
	for k := range images {
		excludes[i] = k
		i++
	}
	if err := chrootarchive.Untar(inTar, repoDir, &archive.TarOptions{ExcludePatterns: excludes}); err != nil {
		return err
	}

	dirs, err := ioutil.ReadDir(repoDir)
	if err != nil {
		return err
	}

	for _, d := range dirs {
		if d.IsDir() {
			if err := s.recursiveLoad(d.Name(), tmpImageDir); err != nil {
				return err
			}
		}
	}

	reposJSONFile, err := os.Open(filepath.Join(tmpImageDir, "repo", "repositories"))
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	defer reposJSONFile.Close()

	repositories := map[string]Repository{}
	if err := json.NewDecoder(reposJSONFile).Decode(&repositories); err != nil {
		return err
	}

	for imageName, tagMap := range repositories {
		for tag, address := range tagMap {
			if err := s.SetLoad(imageName, tag, address, true, outStream); err != nil {
				return err
			}
		}
	}

	return nil
}

func (s *TagStore) recursiveLoad(address, tmpImageDir string) error {
	if _, err := s.LookupImage(address); err != nil {
		logrus.Debugf("Loading %s", address)

		imageJson, err := ioutil.ReadFile(filepath.Join(tmpImageDir, "repo", address, "json"))
		if err != nil {
			logrus.Debugf("Error reading json: %v", err)
			return err
		}

		layer, err := os.Open(filepath.Join(tmpImageDir, "repo", address, "layer.tar"))
		if err != nil {
			logrus.Debugf("Error reading embedded tar: %v", err)
			return err
		}
		img, err := image.NewImgJSON(imageJson)
		if err != nil {
			logrus.Debugf("Error unmarshalling json: %v", err)
			return err
		}
		if err := image.ValidateID(img.ID); err != nil {
			logrus.Debugf("Error validating ID: %v", err)
			return err
		}

		// ensure no two downloads of the same layer happen at the same time
		if c, err := s.poolAdd("pull", "layer:"+img.ID); err != nil {
			if c != nil {
				logrus.Debugf("Image (id: %s) load is already running, waiting: %v", img.ID, err)
				<-c
				return nil
			}

			return err
		}

		defer s.poolRemove("pull", "layer:"+img.ID)

		if img.Parent != "" {
			if !s.graph.Exists(img.Parent) {
				if err := s.recursiveLoad(img.Parent, tmpImageDir); err != nil {
					return err
				}
			}
		}
		if err := s.graph.Register(img, layer); err != nil {
			return err
		}
	}
	logrus.Debugf("Completed processing %s", address)

	return nil
}
