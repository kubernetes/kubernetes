package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-check/check"
)

// This is a heisen-test.  Because the created timestamp of images and the behavior of
// sort is not predictable it doesn't always fail.
func (s *DockerSuite) TestBuildHistory(c *check.C) {
	name := "testbuildhistory"
	_, err := buildImage(name, `FROM busybox
RUN echo "A"
RUN echo "B"
RUN echo "C"
RUN echo "D"
RUN echo "E"
RUN echo "F"
RUN echo "G"
RUN echo "H"
RUN echo "I"
RUN echo "J"
RUN echo "K"
RUN echo "L"
RUN echo "M"
RUN echo "N"
RUN echo "O"
RUN echo "P"
RUN echo "Q"
RUN echo "R"
RUN echo "S"
RUN echo "T"
RUN echo "U"
RUN echo "V"
RUN echo "W"
RUN echo "X"
RUN echo "Y"
RUN echo "Z"`,
		true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "history", "testbuildhistory")
	actualValues := strings.Split(out, "\n")[1:27]
	expectedValues := [26]string{"Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A"}

	for i := 0; i < 26; i++ {
		echoValue := fmt.Sprintf("echo \"%s\"", expectedValues[i])
		actualValue := actualValues[i]

		if !strings.Contains(actualValue, echoValue) {
			c.Fatalf("Expected layer \"%s\", but was: %s", expectedValues[i], actualValue)
		}
	}

}

func (s *DockerSuite) TestHistoryExistentImage(c *check.C) {
	dockerCmd(c, "history", "busybox")
}

func (s *DockerSuite) TestHistoryNonExistentImage(c *check.C) {
	_, _, err := dockerCmdWithError(c, "history", "testHistoryNonExistentImage")
	if err == nil {
		c.Fatal("history on a non-existent image should fail.")
	}
}

func (s *DockerSuite) TestHistoryImageWithComment(c *check.C) {
	name := "testhistoryimagewithcomment"

	// make a image through docker commit <container id> [ -m messages ]

	dockerCmd(c, "run", "--name", name, "busybox", "true")
	dockerCmd(c, "wait", name)

	comment := "This_is_a_comment"
	dockerCmd(c, "commit", "-m="+comment, name, name)

	// test docker history <image id> to check comment messages

	out, _ := dockerCmd(c, "history", name)
	outputTabs := strings.Fields(strings.Split(out, "\n")[1])
	actualValue := outputTabs[len(outputTabs)-1]

	if !strings.Contains(actualValue, comment) {
		c.Fatalf("Expected comments %q, but found %q", comment, actualValue)
	}
}

func (s *DockerSuite) TestHistoryHumanOptionFalse(c *check.C) {
	out, _ := dockerCmd(c, "history", "--human=false", "busybox")
	lines := strings.Split(out, "\n")
	sizeColumnRegex, _ := regexp.Compile("SIZE +")
	indices := sizeColumnRegex.FindStringIndex(lines[0])
	startIndex := indices[0]
	endIndex := indices[1]
	for i := 1; i < len(lines)-1; i++ {
		if endIndex > len(lines[i]) {
			endIndex = len(lines[i])
		}
		sizeString := lines[i][startIndex:endIndex]
		if _, err := strconv.Atoi(strings.TrimSpace(sizeString)); err != nil {
			c.Fatalf("The size '%s' was not an Integer", sizeString)
		}
	}
}

func (s *DockerSuite) TestHistoryHumanOptionTrue(c *check.C) {
	out, _ := dockerCmd(c, "history", "--human=true", "busybox")
	lines := strings.Split(out, "\n")
	sizeColumnRegex, _ := regexp.Compile("SIZE +")
	humanSizeRegex, _ := regexp.Compile("^\\d+.*B$") // Matches human sizes like 10 MB, 3.2 KB, etc
	indices := sizeColumnRegex.FindStringIndex(lines[0])
	startIndex := indices[0]
	endIndex := indices[1]
	for i := 1; i < len(lines)-1; i++ {
		if endIndex > len(lines[i]) {
			endIndex = len(lines[i])
		}
		sizeString := lines[i][startIndex:endIndex]
		if matchSuccess := humanSizeRegex.MatchString(strings.TrimSpace(sizeString)); !matchSuccess {
			c.Fatalf("The size '%s' was not in human format", sizeString)
		}
	}
}
