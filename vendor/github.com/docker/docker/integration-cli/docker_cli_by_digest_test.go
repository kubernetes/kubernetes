package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/docker/docker/utils"
	"github.com/go-check/check"
)

var (
	repoName    = fmt.Sprintf("%v/dockercli/busybox-by-dgst", privateRegistryURL)
	digestRegex = regexp.MustCompile("Digest: ([^\n]+)")
)

func setupImage(c *check.C) (string, error) {
	return setupImageWithTag(c, "latest")
}

func setupImageWithTag(c *check.C, tag string) (string, error) {
	containerName := "busyboxbydigest"

	dockerCmd(c, "run", "-d", "-e", "digest=1", "--name", containerName, "busybox")

	// tag the image to upload it to the private registry
	repoAndTag := utils.ImageReference(repoName, tag)
	if out, _, err := dockerCmdWithError(c, "commit", containerName, repoAndTag); err != nil {
		return "", fmt.Errorf("image tagging failed: %s, %v", out, err)
	}

	// delete the container as we don't need it any more
	if err := deleteContainer(containerName); err != nil {
		return "", err
	}

	// push the image
	out, _, err := dockerCmdWithError(c, "push", repoAndTag)
	if err != nil {
		return "", fmt.Errorf("pushing the image to the private registry has failed: %s, %v", out, err)
	}

	// delete our local repo that we previously tagged
	if rmiout, _, err := dockerCmdWithError(c, "rmi", repoAndTag); err != nil {
		return "", fmt.Errorf("error deleting images prior to real test: %s, %v", rmiout, err)
	}

	// the push output includes "Digest: <digest>", so find that
	matches := digestRegex.FindStringSubmatch(out)
	if len(matches) != 2 {
		return "", fmt.Errorf("unable to parse digest from push output: %s", out)
	}
	pushDigest := matches[1]

	return pushDigest, nil
}

func (s *DockerRegistrySuite) TestPullByTagDisplaysDigest(c *check.C) {
	pushDigest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	// pull from the registry using the tag
	out, _ := dockerCmd(c, "pull", repoName)

	// the pull output includes "Digest: <digest>", so find that
	matches := digestRegex.FindStringSubmatch(out)
	if len(matches) != 2 {
		c.Fatalf("unable to parse digest from pull output: %s", out)
	}
	pullDigest := matches[1]

	// make sure the pushed and pull digests match
	if pushDigest != pullDigest {
		c.Fatalf("push digest %q didn't match pull digest %q", pushDigest, pullDigest)
	}
}

func (s *DockerRegistrySuite) TestPullByDigest(c *check.C) {
	pushDigest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	// pull from the registry using the <name>@<digest> reference
	imageReference := fmt.Sprintf("%s@%s", repoName, pushDigest)
	out, _ := dockerCmd(c, "pull", imageReference)

	// the pull output includes "Digest: <digest>", so find that
	matches := digestRegex.FindStringSubmatch(out)
	if len(matches) != 2 {
		c.Fatalf("unable to parse digest from pull output: %s", out)
	}
	pullDigest := matches[1]

	// make sure the pushed and pull digests match
	if pushDigest != pullDigest {
		c.Fatalf("push digest %q didn't match pull digest %q", pushDigest, pullDigest)
	}
}

func (s *DockerRegistrySuite) TestPullByDigestNoFallback(c *check.C) {
	// pull from the registry using the <name>@<digest> reference
	imageReference := fmt.Sprintf("%s@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", repoName)
	out, _, err := dockerCmdWithError(c, "pull", imageReference)
	if err == nil || !strings.Contains(out, "manifest unknown") {
		c.Fatalf("expected non-zero exit status and correct error message when pulling non-existing image: %s", out)
	}
}

func (s *DockerRegistrySuite) TestCreateByDigest(c *check.C) {
	pushDigest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, pushDigest)

	containerName := "createByDigest"
	out, _ := dockerCmd(c, "create", "--name", containerName, imageReference)

	res, err := inspectField(containerName, "Config.Image")
	if err != nil {
		c.Fatalf("failed to get Config.Image: %s, %v", out, err)
	}
	if res != imageReference {
		c.Fatalf("unexpected Config.Image: %s (expected %s)", res, imageReference)
	}
}

func (s *DockerRegistrySuite) TestRunByDigest(c *check.C) {
	pushDigest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, pushDigest)

	containerName := "runByDigest"
	out, _ := dockerCmd(c, "run", "--name", containerName, imageReference, "sh", "-c", "echo found=$digest")

	foundRegex := regexp.MustCompile("found=([^\n]+)")
	matches := foundRegex.FindStringSubmatch(out)
	if len(matches) != 2 {
		c.Fatalf("error locating expected 'found=1' output: %s", out)
	}
	if matches[1] != "1" {
		c.Fatalf("Expected %q, got %q", "1", matches[1])
	}

	res, err := inspectField(containerName, "Config.Image")
	if err != nil {
		c.Fatalf("failed to get Config.Image: %s, %v", out, err)
	}
	if res != imageReference {
		c.Fatalf("unexpected Config.Image: %s (expected %s)", res, imageReference)
	}
}

func (s *DockerRegistrySuite) TestRemoveImageByDigest(c *check.C) {
	digest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, digest)

	// pull from the registry using the <name>@<digest> reference
	dockerCmd(c, "pull", imageReference)

	// make sure inspect runs ok
	if _, err := inspectField(imageReference, "Id"); err != nil {
		c.Fatalf("failed to inspect image: %v", err)
	}

	// do the delete
	if err := deleteImages(imageReference); err != nil {
		c.Fatalf("unexpected error deleting image: %v", err)
	}

	// try to inspect again - it should error this time
	if _, err := inspectField(imageReference, "Id"); err == nil {
		c.Fatalf("unexpected nil err trying to inspect what should be a non-existent image")
	} else if !strings.Contains(err.Error(), "No such image") {
		c.Fatalf("expected 'No such image' output, got %v", err)
	}
}

func (s *DockerRegistrySuite) TestBuildByDigest(c *check.C) {
	digest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, digest)

	// pull from the registry using the <name>@<digest> reference
	dockerCmd(c, "pull", imageReference)

	// get the image id
	imageID, err := inspectField(imageReference, "Id")
	if err != nil {
		c.Fatalf("error getting image id: %v", err)
	}

	// do the build
	name := "buildbydigest"
	_, err = buildImage(name, fmt.Sprintf(
		`FROM %s
     CMD ["/bin/echo", "Hello World"]`, imageReference),
		true)
	if err != nil {
		c.Fatal(err)
	}

	// get the build's image id
	res, err := inspectField(name, "Config.Image")
	if err != nil {
		c.Fatal(err)
	}
	// make sure they match
	if res != imageID {
		c.Fatalf("Image %s, expected %s", res, imageID)
	}
}

func (s *DockerRegistrySuite) TestTagByDigest(c *check.C) {
	digest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, digest)

	// pull from the registry using the <name>@<digest> reference
	dockerCmd(c, "pull", imageReference)

	// tag it
	tag := "tagbydigest"
	dockerCmd(c, "tag", imageReference, tag)

	expectedID, err := inspectField(imageReference, "Id")
	if err != nil {
		c.Fatalf("error getting original image id: %v", err)
	}

	tagID, err := inspectField(tag, "Id")
	if err != nil {
		c.Fatalf("error getting tagged image id: %v", err)
	}

	if tagID != expectedID {
		c.Fatalf("expected image id %q, got %q", expectedID, tagID)
	}
}

func (s *DockerRegistrySuite) TestListImagesWithoutDigests(c *check.C) {
	digest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	imageReference := fmt.Sprintf("%s@%s", repoName, digest)

	// pull from the registry using the <name>@<digest> reference
	dockerCmd(c, "pull", imageReference)

	out, _ := dockerCmd(c, "images")

	if strings.Contains(out, "DIGEST") {
		c.Fatalf("list output should not have contained DIGEST header: %s", out)
	}

}

func (s *DockerRegistrySuite) TestListImagesWithDigests(c *check.C) {

	// setup image1
	digest1, err := setupImageWithTag(c, "tag1")
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}
	imageReference1 := fmt.Sprintf("%s@%s", repoName, digest1)
	c.Logf("imageReference1 = %s", imageReference1)

	// pull image1 by digest
	dockerCmd(c, "pull", imageReference1)

	// list images
	out, _ := dockerCmd(c, "images", "--digests")

	// make sure repo shown, tag=<none>, digest = $digest1
	re1 := regexp.MustCompile(`\s*` + repoName + `\s*<none>\s*` + digest1 + `\s`)
	if !re1.MatchString(out) {
		c.Fatalf("expected %q: %s", re1.String(), out)
	}

	// setup image2
	digest2, err := setupImageWithTag(c, "tag2")
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}
	imageReference2 := fmt.Sprintf("%s@%s", repoName, digest2)
	c.Logf("imageReference2 = %s", imageReference2)

	// pull image1 by digest
	dockerCmd(c, "pull", imageReference1)

	// pull image2 by digest
	dockerCmd(c, "pull", imageReference2)

	// list images
	out, _ = dockerCmd(c, "images", "--digests")

	// make sure repo shown, tag=<none>, digest = $digest1
	if !re1.MatchString(out) {
		c.Fatalf("expected %q: %s", re1.String(), out)
	}

	// make sure repo shown, tag=<none>, digest = $digest2
	re2 := regexp.MustCompile(`\s*` + repoName + `\s*<none>\s*` + digest2 + `\s`)
	if !re2.MatchString(out) {
		c.Fatalf("expected %q: %s", re2.String(), out)
	}

	// pull tag1
	dockerCmd(c, "pull", repoName+":tag1")

	// list images
	out, _ = dockerCmd(c, "images", "--digests")

	// make sure image 1 has repo, tag, <none> AND repo, <none>, digest
	reWithTag1 := regexp.MustCompile(`\s*` + repoName + `\s*tag1\s*<none>\s`)
	reWithDigest1 := regexp.MustCompile(`\s*` + repoName + `\s*<none>\s*` + digest1 + `\s`)
	if !reWithTag1.MatchString(out) {
		c.Fatalf("expected %q: %s", reWithTag1.String(), out)
	}
	if !reWithDigest1.MatchString(out) {
		c.Fatalf("expected %q: %s", reWithDigest1.String(), out)
	}
	// make sure image 2 has repo, <none>, digest
	if !re2.MatchString(out) {
		c.Fatalf("expected %q: %s", re2.String(), out)
	}

	// pull tag 2
	dockerCmd(c, "pull", repoName+":tag2")

	// list images
	out, _ = dockerCmd(c, "images", "--digests")

	// make sure image 1 has repo, tag, digest
	if !reWithTag1.MatchString(out) {
		c.Fatalf("expected %q: %s", re1.String(), out)
	}

	// make sure image 2 has repo, tag, digest
	reWithTag2 := regexp.MustCompile(`\s*` + repoName + `\s*tag2\s*<none>\s`)
	reWithDigest2 := regexp.MustCompile(`\s*` + repoName + `\s*<none>\s*` + digest2 + `\s`)
	if !reWithTag2.MatchString(out) {
		c.Fatalf("expected %q: %s", reWithTag2.String(), out)
	}
	if !reWithDigest2.MatchString(out) {
		c.Fatalf("expected %q: %s", reWithDigest2.String(), out)
	}

	// list images
	out, _ = dockerCmd(c, "images", "--digests")

	// make sure image 1 has repo, tag, digest
	if !reWithTag1.MatchString(out) {
		c.Fatalf("expected %q: %s", re1.String(), out)
	}
	// make sure image 2 has repo, tag, digest
	if !reWithTag2.MatchString(out) {
		c.Fatalf("expected %q: %s", re2.String(), out)
	}
	// make sure busybox has tag, but not digest
	busyboxRe := regexp.MustCompile(`\s*busybox\s*latest\s*<none>\s`)
	if !busyboxRe.MatchString(out) {
		c.Fatalf("expected %q: %s", busyboxRe.String(), out)
	}
}

func (s *DockerRegistrySuite) TestDeleteImageByIDOnlyPulledByDigest(c *check.C) {
	pushDigest, err := setupImage(c)
	if err != nil {
		c.Fatalf("error setting up image: %v", err)
	}

	// pull from the registry using the <name>@<digest> reference
	imageReference := fmt.Sprintf("%s@%s", repoName, pushDigest)
	dockerCmd(c, "pull", imageReference)
	// just in case...

	imageID, err := inspectField(imageReference, "Id")
	if err != nil {
		c.Fatalf("error inspecting image id: %v", err)
	}

	dockerCmd(c, "rmi", imageID)
}
