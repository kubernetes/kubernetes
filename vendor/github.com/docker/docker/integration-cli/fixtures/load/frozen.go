package load

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
)

var frozenImgDir = "/docker-frozen-images"

// FrozenImagesLinux loads the frozen image set for the integration suite
// If the images are not available locally it will download them
// TODO: This loads whatever is in the frozen image dir, regardless of what
// images were passed in. If the images need to be downloaded, then it will respect
// the passed in images
func FrozenImagesLinux(dockerBinary string, images ...string) error {
	imgNS := os.Getenv("TEST_IMAGE_NAMESPACE")
	var loadImages []struct{ srcName, destName string }
	for _, img := range images {
		if err := exec.Command(dockerBinary, "inspect", "--type=image", img).Run(); err != nil {
			srcName := img
			// hello-world:latest gets re-tagged as hello-world:frozen
			// there are some tests that use hello-world:latest specifically so it pulls
			// the image and hello-world:frozen is used for when we just want a super
			// small image
			if img == "hello-world:frozen" {
				srcName = "hello-world:latest"
			}
			if imgNS != "" {
				srcName = imgNS + "/" + srcName
			}
			loadImages = append(loadImages, struct{ srcName, destName string }{
				srcName:  srcName,
				destName: img,
			})
		}
	}
	if len(loadImages) == 0 {
		// everything is loaded, we're done
		return nil
	}

	fi, err := os.Stat(frozenImgDir)
	if err != nil || !fi.IsDir() {
		srcImages := make([]string, 0, len(loadImages))
		for _, img := range loadImages {
			srcImages = append(srcImages, img.srcName)
		}
		if err := pullImages(dockerBinary, srcImages); err != nil {
			return errors.Wrap(err, "error pulling image list")
		}
	} else {
		if err := loadFrozenImages(dockerBinary); err != nil {
			return err
		}
	}

	for _, img := range loadImages {
		if img.srcName != img.destName {
			if out, err := exec.Command(dockerBinary, "tag", img.srcName, img.destName).CombinedOutput(); err != nil {
				return errors.Errorf("%v: %s", err, string(out))
			}
			if out, err := exec.Command(dockerBinary, "rmi", img.srcName).CombinedOutput(); err != nil {
				return errors.Errorf("%v: %s", err, string(out))
			}
		}
	}
	return nil
}

func loadFrozenImages(dockerBinary string) error {
	tar, err := exec.LookPath("tar")
	if err != nil {
		return errors.Wrap(err, "could not find tar binary")
	}
	tarCmd := exec.Command(tar, "-cC", frozenImgDir, ".")
	out, err := tarCmd.StdoutPipe()
	if err != nil {
		return errors.Wrap(err, "error getting stdout pipe for tar command")
	}

	errBuf := bytes.NewBuffer(nil)
	tarCmd.Stderr = errBuf
	tarCmd.Start()
	defer tarCmd.Wait()

	cmd := exec.Command(dockerBinary, "load")
	cmd.Stdin = out
	if out, err := cmd.CombinedOutput(); err != nil {
		return errors.Errorf("%v: %s", err, string(out))
	}
	return nil
}

func pullImages(dockerBinary string, images []string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return errors.Wrap(err, "error getting path to dockerfile")
	}
	dockerfile := os.Getenv("DOCKERFILE")
	if dockerfile == "" {
		dockerfile = "Dockerfile"
	}
	dockerfilePath := filepath.Join(filepath.Dir(filepath.Clean(cwd)), dockerfile)
	pullRefs, err := readFrozenImageList(dockerfilePath, images)
	if err != nil {
		return errors.Wrap(err, "error reading frozen image list")
	}

	var wg sync.WaitGroup
	chErr := make(chan error, len(images))
	for tag, ref := range pullRefs {
		wg.Add(1)
		go func(tag, ref string) {
			defer wg.Done()
			if out, err := exec.Command(dockerBinary, "pull", ref).CombinedOutput(); err != nil {
				chErr <- errors.Errorf("%v: %s", string(out), err)
				return
			}
			if out, err := exec.Command(dockerBinary, "tag", ref, tag).CombinedOutput(); err != nil {
				chErr <- errors.Errorf("%v: %s", string(out), err)
				return
			}
			if out, err := exec.Command(dockerBinary, "rmi", ref).CombinedOutput(); err != nil {
				chErr <- errors.Errorf("%v: %s", string(out), err)
				return
			}
		}(tag, ref)
	}
	wg.Wait()
	close(chErr)
	return <-chErr
}

func readFrozenImageList(dockerfilePath string, images []string) (map[string]string, error) {
	f, err := os.Open(dockerfilePath)
	if err != nil {
		return nil, errors.Wrap(err, "error reading dockerfile")
	}
	defer f.Close()
	ls := make(map[string]string)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.Fields(scanner.Text())
		if len(line) < 3 {
			continue
		}
		if !(line[0] == "RUN" && line[1] == "./contrib/download-frozen-image-v2.sh") {
			continue
		}

		frozenImgDir = line[2]
		if line[2] == frozenImgDir {
			frozenImgDir = filepath.Join(os.Getenv("DEST"), "frozen-images")
		}

		for scanner.Scan() {
			img := strings.TrimSpace(scanner.Text())
			img = strings.TrimSuffix(img, "\\")
			img = strings.TrimSpace(img)
			split := strings.Split(img, "@")
			if len(split) < 2 {
				break
			}

			for _, i := range images {
				if split[0] == i {
					ls[i] = img
					break
				}
			}
		}
	}
	return ls, nil
}
