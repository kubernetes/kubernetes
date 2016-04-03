package main

import (
	"encoding/json"
	"net/http"
	"net/url"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestApiImagesFilter(c *check.C) {
	name := "utest:tag1"
	name2 := "utest/docker:tag2"
	name3 := "utest:5000/docker:tag3"
	for _, n := range []string{name, name2, name3} {
		dockerCmd(c, "tag", "busybox", n)
	}
	type image types.Image
	getImages := func(filter string) []image {
		v := url.Values{}
		v.Set("filter", filter)
		status, b, err := sockRequest("GET", "/images/json?"+v.Encode(), nil)
		c.Assert(status, check.Equals, http.StatusOK)
		c.Assert(err, check.IsNil)

		var images []image
		if err := json.Unmarshal(b, &images); err != nil {
			c.Fatal(err)
		}

		return images
	}

	errMsg := "incorrect number of matches returned"
	if images := getImages("utest*/*"); len(images[0].RepoTags) != 2 {
		c.Fatal(errMsg)
	}
	if images := getImages("utest"); len(images[0].RepoTags) != 1 {
		c.Fatal(errMsg)
	}
	if images := getImages("utest*"); len(images[0].RepoTags) != 1 {
		c.Fatal(errMsg)
	}
	if images := getImages("*5000*/*"); len(images[0].RepoTags) != 1 {
		c.Fatal(errMsg)
	}
}

func (s *DockerSuite) TestApiImagesSaveAndLoad(c *check.C) {
	testRequires(c, Network)
	out, err := buildImage("saveandload", "FROM hello-world\nENV FOO bar", false)
	if err != nil {
		c.Fatal(err)
	}
	id := strings.TrimSpace(out)

	res, body, err := sockRequestRaw("GET", "/images/"+id+"/get", nil, "")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	defer body.Close()

	dockerCmd(c, "rmi", id)

	res, loadBody, err := sockRequestRaw("POST", "/images/load", body, "application/x-tar")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	defer loadBody.Close()

	inspectOut, _ := dockerCmd(c, "inspect", "--format='{{ .Id }}'", id)
	if strings.TrimSpace(string(inspectOut)) != id {
		c.Fatal("load did not work properly")
	}
}

func (s *DockerSuite) TestApiImagesDelete(c *check.C) {
	testRequires(c, Network)
	name := "test-api-images-delete"
	out, err := buildImage(name, "FROM hello-world\nENV FOO bar", false)
	if err != nil {
		c.Fatal(err)
	}
	id := strings.TrimSpace(out)

	dockerCmd(c, "tag", name, "test:tag1")

	status, _, err := sockRequest("DELETE", "/images/"+id, nil)
	c.Assert(status, check.Equals, http.StatusConflict)
	c.Assert(err, check.IsNil)

	status, _, err = sockRequest("DELETE", "/images/test:noexist", nil)
	c.Assert(status, check.Equals, http.StatusNotFound) //Status Codes:404 – no such image
	c.Assert(err, check.IsNil)

	status, _, err = sockRequest("DELETE", "/images/test:tag1", nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)
}

func (s *DockerSuite) TestApiImagesHistory(c *check.C) {
	testRequires(c, Network)
	name := "test-api-images-history"
	out, err := buildImage(name, "FROM hello-world\nENV FOO bar", false)
	c.Assert(err, check.IsNil)

	id := strings.TrimSpace(out)

	status, body, err := sockRequest("GET", "/images/"+id+"/history", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var historydata []types.ImageHistory
	if err = json.Unmarshal(body, &historydata); err != nil {
		c.Fatalf("Error on unmarshal: %s", err)
	}

	c.Assert(len(historydata), check.Not(check.Equals), 0)
	c.Assert(historydata[0].Tags[0], check.Equals, "test-api-images-history:latest")
}
