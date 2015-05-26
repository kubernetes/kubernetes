// +build ignore

// Simple tool to create an archive stream from an old and new directory
//
// By default it will stream the comparison of two temporary directories with junk files
package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"

	"github.com/fsouza/go-dockerclient/vendor/github.com/Sirupsen/logrus"
	"github.com/fsouza/go-dockerclient/vendor/github.com/docker/docker/pkg/archive"
)

var (
	flDebug  = flag.Bool("D", false, "debugging output")
	flNewDir = flag.String("newdir", "", "")
	flOldDir = flag.String("olddir", "", "")
	log      = logrus.New()
)

func main() {
	flag.Usage = func() {
		fmt.Println("Produce a tar from comparing two directory paths. By default a demo tar is created of around 200 files (including hardlinks)")
		fmt.Printf("%s [OPTIONS]\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	log.Out = os.Stderr
	if (len(os.Getenv("DEBUG")) > 0) || *flDebug {
		logrus.SetLevel(logrus.DebugLevel)
	}
	var newDir, oldDir string

	if len(*flNewDir) == 0 {
		var err error
		newDir, err = ioutil.TempDir("", "docker-test-newDir")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(newDir)
		if _, err := prepareUntarSourceDirectory(100, newDir, true); err != nil {
			log.Fatal(err)
		}
	} else {
		newDir = *flNewDir
	}

	if len(*flOldDir) == 0 {
		oldDir, err := ioutil.TempDir("", "docker-test-oldDir")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(oldDir)
	} else {
		oldDir = *flOldDir
	}

	changes, err := archive.ChangesDirs(newDir, oldDir)
	if err != nil {
		log.Fatal(err)
	}

	a, err := archive.ExportChanges(newDir, changes)
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	i, err := io.Copy(os.Stdout, a)
	if err != nil && err != io.EOF {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "wrote archive of %d bytes", i)
}

func prepareUntarSourceDirectory(numberOfFiles int, targetPath string, makeLinks bool) (int, error) {
	fileData := []byte("fooo")
	for n := 0; n < numberOfFiles; n++ {
		fileName := fmt.Sprintf("file-%d", n)
		if err := ioutil.WriteFile(path.Join(targetPath, fileName), fileData, 0700); err != nil {
			return 0, err
		}
		if makeLinks {
			if err := os.Link(path.Join(targetPath, fileName), path.Join(targetPath, fileName+"-link")); err != nil {
				return 0, err
			}
		}
	}
	totalSize := numberOfFiles * len(fileData)
	return totalSize, nil
}
