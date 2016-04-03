package main

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/docker/docker/pkg/stringid"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestImagesEnsureImageIsListed(c *check.C) {
	out, _ := dockerCmd(c, "images")
	if !strings.Contains(out, "busybox") {
		c.Fatal("images should've listed busybox")
	}
}

func (s *DockerSuite) TestImagesOrderedByCreationDate(c *check.C) {
	id1, err := buildImage("order:test_a",
		`FROM scratch
		MAINTAINER dockerio1`, true)
	if err != nil {
		c.Fatal(err)
	}
	time.Sleep(time.Second)
	id2, err := buildImage("order:test_c",
		`FROM scratch
		MAINTAINER dockerio2`, true)
	if err != nil {
		c.Fatal(err)
	}
	time.Sleep(time.Second)
	id3, err := buildImage("order:test_b",
		`FROM scratch
		MAINTAINER dockerio3`, true)
	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "images", "-q", "--no-trunc")
	imgs := strings.Split(out, "\n")
	if imgs[0] != id3 {
		c.Fatalf("First image must be %s, got %s", id3, imgs[0])
	}
	if imgs[1] != id2 {
		c.Fatalf("Second image must be %s, got %s", id2, imgs[1])
	}
	if imgs[2] != id1 {
		c.Fatalf("Third image must be %s, got %s", id1, imgs[2])
	}
}

func (s *DockerSuite) TestImagesErrorWithInvalidFilterNameTest(c *check.C) {
	out, _, err := dockerCmdWithError(c, "images", "-f", "FOO=123")
	if err == nil || !strings.Contains(out, "Invalid filter") {
		c.Fatalf("error should occur when listing images with invalid filter name FOO, %s", out)
	}
}

func (s *DockerSuite) TestImagesFilterLabel(c *check.C) {
	imageName1 := "images_filter_test1"
	imageName2 := "images_filter_test2"
	imageName3 := "images_filter_test3"
	image1ID, err := buildImage(imageName1,
		`FROM scratch
		 LABEL match me`, true)
	if err != nil {
		c.Fatal(err)
	}

	image2ID, err := buildImage(imageName2,
		`FROM scratch
		 LABEL match="me too"`, true)
	if err != nil {
		c.Fatal(err)
	}

	image3ID, err := buildImage(imageName3,
		`FROM scratch
		 LABEL nomatch me`, true)
	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "images", "--no-trunc", "-q", "-f", "label=match")
	out = strings.TrimSpace(out)
	if (!strings.Contains(out, image1ID) && !strings.Contains(out, image2ID)) || strings.Contains(out, image3ID) {
		c.Fatalf("Expected ids %s,%s got %s", image1ID, image2ID, out)
	}

	out, _ = dockerCmd(c, "images", "--no-trunc", "-q", "-f", "label=match=me too")
	out = strings.TrimSpace(out)
	if out != image2ID {
		c.Fatalf("Expected %s got %s", image2ID, out)
	}
}

func (s *DockerSuite) TestImagesFilterSpaceTrimCase(c *check.C) {
	imageName := "images_filter_test"
	buildImage(imageName,
		`FROM scratch
		 RUN touch /test/foo
		 RUN touch /test/bar
		 RUN touch /test/baz`, true)

	filters := []string{
		"dangling=true",
		"Dangling=true",
		" dangling=true",
		"dangling=true ",
		"dangling = true",
	}

	imageListings := make([][]string, 5, 5)
	for idx, filter := range filters {
		out, _ := dockerCmd(c, "images", "-q", "-f", filter)
		listing := strings.Split(out, "\n")
		sort.Strings(listing)
		imageListings[idx] = listing
	}

	for idx, listing := range imageListings {
		if idx < 4 && !reflect.DeepEqual(listing, imageListings[idx+1]) {
			for idx, errListing := range imageListings {
				fmt.Printf("out %d", idx)
				for _, image := range errListing {
					fmt.Print(image)
				}
				fmt.Print("")
			}
			c.Fatalf("All output must be the same")
		}
	}
}

func (s *DockerSuite) TestImagesEnsureDanglingImageOnlyListedOnce(c *check.C) {
	// create container 1
	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")
	containerId1 := strings.TrimSpace(out)

	// tag as foobox
	out, _ = dockerCmd(c, "commit", containerId1, "foobox")
	imageId := stringid.TruncateID(strings.TrimSpace(out))

	// overwrite the tag, making the previous image dangling
	dockerCmd(c, "tag", "-f", "busybox", "foobox")

	out, _ = dockerCmd(c, "images", "-q", "-f", "dangling=true")
	if e, a := 1, strings.Count(out, imageId); e != a {
		c.Fatalf("expected 1 dangling image, got %d: %s", a, out)
	}
}
