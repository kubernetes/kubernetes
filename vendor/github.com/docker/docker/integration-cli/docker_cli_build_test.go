package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/docker/docker/builder/command"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/stringutils"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestBuildJSONEmptyRun(c *check.C) {
	name := "testbuildjsonemptyrun"

	_, err := buildImage(
		name,
		`
    FROM busybox
    RUN []
    `,
		true)

	if err != nil {
		c.Fatal("error when dealing with a RUN statement with empty JSON array")
	}

}

func (s *DockerSuite) TestBuildEmptyWhitespace(c *check.C) {
	name := "testbuildemptywhitespace"

	_, err := buildImage(
		name,
		`
    FROM busybox
    COPY
      quux \
      bar
    `,
		true)

	if err == nil {
		c.Fatal("no error when dealing with a COPY statement with no content on the same line")
	}

}

func (s *DockerSuite) TestBuildShCmdJSONEntrypoint(c *check.C) {
	name := "testbuildshcmdjsonentrypoint"

	_, err := buildImage(
		name,
		`
    FROM busybox
    ENTRYPOINT ["/bin/echo"]
    CMD echo test
    `,
		true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "--rm", name)

	if strings.TrimSpace(out) != "/bin/sh -c echo test" {
		c.Fatal("CMD did not contain /bin/sh -c")
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementUser(c *check.C) {
	name := "testbuildenvironmentreplacement"

	_, err := buildImage(name, `
  FROM scratch
  ENV user foo
  USER ${user}
  `, true)
	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.User")
	if err != nil {
		c.Fatal(err)
	}

	if res != `"foo"` {
		c.Fatal("User foo from environment not in Config.User on image")
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementVolume(c *check.C) {
	name := "testbuildenvironmentreplacement"

	_, err := buildImage(name, `
  FROM scratch
  ENV volume /quux
  VOLUME ${volume}
  `, true)
	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.Volumes")
	if err != nil {
		c.Fatal(err)
	}

	var volumes map[string]interface{}

	if err := json.Unmarshal([]byte(res), &volumes); err != nil {
		c.Fatal(err)
	}

	if _, ok := volumes["/quux"]; !ok {
		c.Fatal("Volume /quux from environment not in Config.Volumes on image")
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementExpose(c *check.C) {
	name := "testbuildenvironmentreplacement"

	_, err := buildImage(name, `
  FROM scratch
  ENV port 80
  EXPOSE ${port}
  `, true)
	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}

	var exposedPorts map[string]interface{}

	if err := json.Unmarshal([]byte(res), &exposedPorts); err != nil {
		c.Fatal(err)
	}

	if _, ok := exposedPorts["80/tcp"]; !ok {
		c.Fatal("Exposed port 80 from environment not in Config.ExposedPorts on image")
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementWorkdir(c *check.C) {
	name := "testbuildenvironmentreplacement"

	_, err := buildImage(name, `
  FROM busybox
  ENV MYWORKDIR /work
  RUN mkdir ${MYWORKDIR}
  WORKDIR ${MYWORKDIR}
  `, true)

	if err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementAddCopy(c *check.C) {
	name := "testbuildenvironmentreplacement"

	ctx, err := fakeContext(`
  FROM scratch
  ENV baz foo
  ENV quux bar
  ENV dot .
  ENV fee fff
  ENV gee ggg

  ADD ${baz} ${dot}
  COPY ${quux} ${dot}
  ADD ${zzz:-${fee}} ${dot}
  COPY ${zzz:-${gee}} ${dot}
  `,
		map[string]string{
			"foo": "test1",
			"bar": "test2",
			"fff": "test3",
			"ggg": "test4",
		})

	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildEnvironmentReplacementEnv(c *check.C) {
	name := "testbuildenvironmentreplacement"

	_, err := buildImage(name,
		`
  FROM busybox
  ENV foo zzz
  ENV bar ${foo}
  ENV abc1='$foo'
  ENV env1=$foo env2=${foo} env3="$foo" env4="${foo}"
  RUN [ "$abc1" = '$foo' ] && (echo "$abc1" | grep -q foo)
  ENV abc2="\$foo"
  RUN [ "$abc2" = '$foo' ] && (echo "$abc2" | grep -q foo)
  ENV abc3 '$foo'
  RUN [ "$abc3" = '$foo' ] && (echo "$abc3" | grep -q foo)
  ENV abc4 "\$foo"
  RUN [ "$abc4" = '$foo' ] && (echo "$abc4" | grep -q foo)
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.Env")
	if err != nil {
		c.Fatal(err)
	}

	envResult := []string{}

	if err = unmarshalJSON([]byte(res), &envResult); err != nil {
		c.Fatal(err)
	}

	found := false
	envCount := 0

	for _, env := range envResult {
		parts := strings.SplitN(env, "=", 2)
		if parts[0] == "bar" {
			found = true
			if parts[1] != "zzz" {
				c.Fatalf("Could not find replaced var for env `bar`: got %q instead of `zzz`", parts[1])
			}
		} else if strings.HasPrefix(parts[0], "env") {
			envCount++
			if parts[1] != "zzz" {
				c.Fatalf("%s should be 'foo' but instead its %q", parts[0], parts[1])
			}
		} else if strings.HasPrefix(parts[0], "env") {
			envCount++
			if parts[1] != "foo" {
				c.Fatalf("%s should be 'foo' but instead its %q", parts[0], parts[1])
			}
		}
	}

	if !found {
		c.Fatal("Never found the `bar` env variable")
	}

	if envCount != 4 {
		c.Fatalf("Didn't find all env vars - only saw %d\n%s", envCount, envResult)
	}

}

func (s *DockerSuite) TestBuildHandleEscapes(c *check.C) {
	name := "testbuildhandleescapes"

	_, err := buildImage(name,
		`
  FROM scratch
  ENV FOO bar
  VOLUME ${FOO}
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	var result map[string]map[string]struct{}

	res, err := inspectFieldJSON(name, "Config.Volumes")
	if err != nil {
		c.Fatal(err)
	}

	if err = unmarshalJSON([]byte(res), &result); err != nil {
		c.Fatal(err)
	}

	if _, ok := result["bar"]; !ok {
		c.Fatal("Could not find volume bar set from env foo in volumes table")
	}

	deleteImages(name)

	_, err = buildImage(name,
		`
  FROM scratch
  ENV FOO bar
  VOLUME \${FOO}
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	res, err = inspectFieldJSON(name, "Config.Volumes")
	if err != nil {
		c.Fatal(err)
	}

	if err = unmarshalJSON([]byte(res), &result); err != nil {
		c.Fatal(err)
	}

	if _, ok := result["${FOO}"]; !ok {
		c.Fatal("Could not find volume ${FOO} set from env foo in volumes table")
	}

	deleteImages(name)

	// this test in particular provides *7* backslashes and expects 6 to come back.
	// Like above, the first escape is swallowed and the rest are treated as
	// literals, this one is just less obvious because of all the character noise.

	_, err = buildImage(name,
		`
  FROM scratch
  ENV FOO bar
  VOLUME \\\\\\\${FOO}
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	res, err = inspectFieldJSON(name, "Config.Volumes")
	if err != nil {
		c.Fatal(err)
	}

	if err = unmarshalJSON([]byte(res), &result); err != nil {
		c.Fatal(err)
	}

	if _, ok := result[`\\\${FOO}`]; !ok {
		c.Fatal(`Could not find volume \\\${FOO} set from env foo in volumes table`, result)
	}

}

func (s *DockerSuite) TestBuildOnBuildLowercase(c *check.C) {
	name := "testbuildonbuildlowercase"
	name2 := "testbuildonbuildlowercase2"

	_, err := buildImage(name,
		`
  FROM busybox
  onbuild run echo quux
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	_, out, err := buildImageWithOut(name2, fmt.Sprintf(`
  FROM %s
  `, name), true)

	if err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(out, "quux") {
		c.Fatalf("Did not receive the expected echo text, got %s", out)
	}

	if strings.Contains(out, "ONBUILD ONBUILD") {
		c.Fatalf("Got an ONBUILD ONBUILD error with no error: got %s", out)
	}

}

func (s *DockerSuite) TestBuildEnvEscapes(c *check.C) {
	name := "testbuildenvescapes"
	_, err := buildImage(name,
		`
    FROM busybox
    ENV TEST foo
    CMD echo \$
    `,
		true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "-t", name)

	if strings.TrimSpace(out) != "$" {
		c.Fatalf("Env TEST was not overwritten with bar when foo was supplied to dockerfile: was %q", strings.TrimSpace(out))
	}

}

func (s *DockerSuite) TestBuildEnvOverwrite(c *check.C) {
	name := "testbuildenvoverwrite"

	_, err := buildImage(name,
		`
    FROM busybox
    ENV TEST foo
    CMD echo ${TEST}
    `,
		true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "-e", "TEST=bar", "-t", name)

	if strings.TrimSpace(out) != "bar" {
		c.Fatalf("Env TEST was not overwritten with bar when foo was supplied to dockerfile: was %q", strings.TrimSpace(out))
	}

}

func (s *DockerSuite) TestBuildOnBuildForbiddenMaintainerInSourceImage(c *check.C) {
	name := "testbuildonbuildforbiddenmaintainerinsourceimage"

	out, _ := dockerCmd(c, "create", "busybox", "true")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "commit", "--run", "{\"OnBuild\":[\"MAINTAINER docker.io\"]}", cleanedContainerID, "onbuild")

	_, err := buildImage(name,
		`FROM onbuild`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "maintainer isn't allowed as an ONBUILD trigger") {
			c.Fatalf("Wrong error %v, must be about MAINTAINER and ONBUILD in source image", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}

}

func (s *DockerSuite) TestBuildOnBuildForbiddenFromInSourceImage(c *check.C) {
	name := "testbuildonbuildforbiddenfrominsourceimage"

	out, _ := dockerCmd(c, "create", "busybox", "true")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "commit", "--run", "{\"OnBuild\":[\"FROM busybox\"]}", cleanedContainerID, "onbuild")

	_, err := buildImage(name,
		`FROM onbuild`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "from isn't allowed as an ONBUILD trigger") {
			c.Fatalf("Wrong error %v, must be about FROM and ONBUILD in source image", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}

}

func (s *DockerSuite) TestBuildOnBuildForbiddenChainedInSourceImage(c *check.C) {
	name := "testbuildonbuildforbiddenchainedinsourceimage"

	out, _ := dockerCmd(c, "create", "busybox", "true")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "commit", "--run", "{\"OnBuild\":[\"ONBUILD RUN ls\"]}", cleanedContainerID, "onbuild")

	_, err := buildImage(name,
		`FROM onbuild`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed") {
			c.Fatalf("Wrong error %v, must be about chaining ONBUILD in source image", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}

}

func (s *DockerSuite) TestBuildOnBuildCmdEntrypointJSON(c *check.C) {
	name1 := "onbuildcmd"
	name2 := "onbuildgenerated"

	_, err := buildImage(name1, `
FROM busybox
ONBUILD CMD ["hello world"]
ONBUILD ENTRYPOINT ["echo"]
ONBUILD RUN ["true"]`,
		false)

	if err != nil {
		c.Fatal(err)
	}

	_, err = buildImage(name2, fmt.Sprintf(`FROM %s`, name1), false)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "-t", name2)

	if !regexp.MustCompile(`(?m)^hello world`).MatchString(out) {
		c.Fatal("did not get echo output from onbuild", out)
	}

}

func (s *DockerSuite) TestBuildOnBuildEntrypointJSON(c *check.C) {
	name1 := "onbuildcmd"
	name2 := "onbuildgenerated"

	_, err := buildImage(name1, `
FROM busybox
ONBUILD ENTRYPOINT ["echo"]`,
		false)

	if err != nil {
		c.Fatal(err)
	}

	_, err = buildImage(name2, fmt.Sprintf("FROM %s\nCMD [\"hello world\"]\n", name1), false)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "-t", name2)

	if !regexp.MustCompile(`(?m)^hello world`).MatchString(out) {
		c.Fatal("got malformed output from onbuild", out)
	}

}

func (s *DockerSuite) TestBuildCacheAdd(c *check.C) {
	name := "testbuildtwoimageswithadd"
	server, err := fakeStorage(map[string]string{
		"robots.txt": "hello",
		"index.html": "world",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	if _, err := buildImage(name,
		fmt.Sprintf(`FROM scratch
		ADD %s/robots.txt /`, server.URL()),
		true); err != nil {
		c.Fatal(err)
	}
	if err != nil {
		c.Fatal(err)
	}
	deleteImages(name)
	_, out, err := buildImageWithOut(name,
		fmt.Sprintf(`FROM scratch
		ADD %s/index.html /`, server.URL()),
		true)
	if err != nil {
		c.Fatal(err)
	}
	if strings.Contains(out, "Using cache") {
		c.Fatal("2nd build used cache on ADD, it shouldn't")
	}

}

func (s *DockerSuite) TestBuildLastModified(c *check.C) {
	name := "testbuildlastmodified"

	server, err := fakeStorage(map[string]string{
		"file": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	var out, out2 string

	dFmt := `FROM busybox
ADD %s/file /
RUN ls -le /file`

	dockerfile := fmt.Sprintf(dFmt, server.URL())

	if _, out, err = buildImageWithOut(name, dockerfile, false); err != nil {
		c.Fatal(err)
	}

	originMTime := regexp.MustCompile(`root.*/file.*\n`).FindString(out)
	// Make sure our regexp is correct
	if strings.Index(originMTime, "/file") < 0 {
		c.Fatalf("Missing ls info on 'file':\n%s", out)
	}

	// Build it again and make sure the mtime of the file didn't change.
	// Wait a few seconds to make sure the time changed enough to notice
	time.Sleep(2 * time.Second)

	if _, out2, err = buildImageWithOut(name, dockerfile, false); err != nil {
		c.Fatal(err)
	}

	newMTime := regexp.MustCompile(`root.*/file.*\n`).FindString(out2)
	if newMTime != originMTime {
		c.Fatalf("MTime changed:\nOrigin:%s\nNew:%s", originMTime, newMTime)
	}

	// Now 'touch' the file and make sure the timestamp DID change this time
	// Create a new fakeStorage instead of just using Add() to help windows
	server, err = fakeStorage(map[string]string{
		"file": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	dockerfile = fmt.Sprintf(dFmt, server.URL())

	if _, out2, err = buildImageWithOut(name, dockerfile, false); err != nil {
		c.Fatal(err)
	}

	newMTime = regexp.MustCompile(`root.*/file.*\n`).FindString(out2)
	if newMTime == originMTime {
		c.Fatalf("MTime didn't change:\nOrigin:%s\nNew:%s", originMTime, newMTime)
	}

}

func (s *DockerSuite) TestBuildSixtySteps(c *check.C) {
	name := "foobuildsixtysteps"
	ctx, err := fakeContext("FROM scratch\n"+strings.Repeat("ADD foo /\n", 60),
		map[string]string{
			"foo": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddSingleFileToRoot(c *check.C) {
	name := "testaddimg"
	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio /exists
ADD test_file /
RUN [ $(ls -l /test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_file | awk '{print $1}') = '%s' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`, expectedFileChmod),
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

// Issue #3960: "ADD src ." hangs
func (s *DockerSuite) TestBuildAddSingleFileToWorkdir(c *check.C) {
	name := "testaddsinglefiletoworkdir"
	ctx, err := fakeContext(`FROM busybox
ADD test_file .`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	errChan := make(chan error)
	go func() {
		_, err := buildImageFromContext(name, ctx, true)
		errChan <- err
		close(errChan)
	}()
	select {
	case <-time.After(5 * time.Second):
		c.Fatal("Build with adding to workdir timed out")
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	}
}

func (s *DockerSuite) TestBuildAddSingleFileToExistDir(c *check.C) {
	name := "testaddsinglefiletoexistdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN mkdir /exists
RUN touch /exists/exists_file
RUN chown -R dockerio.dockerio /exists
ADD test_file /exists/
RUN [ $(ls -l / | grep exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists/exists_file | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyAddMultipleFiles(c *check.C) {
	server, err := fakeStorage(map[string]string{
		"robots.txt": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	name := "testcopymultiplefilestofile"
	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN mkdir /exists
RUN touch /exists/exists_file
RUN chown -R dockerio.dockerio /exists
COPY test_file1 test_file2 /exists/
ADD test_file3 test_file4 %s/robots.txt /exists/
RUN [ $(ls -l / | grep exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/test_file1 | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists/test_file2 | awk '{print $3":"$4}') = 'root:root' ]

RUN [ $(ls -l /exists/test_file3 | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists/test_file4 | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists/robots.txt | awk '{print $3":"$4}') = 'root:root' ]

RUN [ $(ls -l /exists/exists_file | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
`, server.URL()),
		map[string]string{
			"test_file1": "test1",
			"test_file2": "test2",
			"test_file3": "test3",
			"test_file4": "test4",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddMultipleFilesToFile(c *check.C) {
	name := "testaddmultiplefilestofile"
	ctx, err := fakeContext(`FROM scratch
	ADD file1.txt file2.txt test
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using ADD with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildJSONAddMultipleFilesToFile(c *check.C) {
	name := "testjsonaddmultiplefilestofile"
	ctx, err := fakeContext(`FROM scratch
	ADD ["file1.txt", "file2.txt", "test"]
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using ADD with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildAddMultipleFilesToFileWild(c *check.C) {
	name := "testaddmultiplefilestofilewild"
	ctx, err := fakeContext(`FROM scratch
	ADD file*.txt test
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using ADD with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildJSONAddMultipleFilesToFileWild(c *check.C) {
	name := "testjsonaddmultiplefilestofilewild"
	ctx, err := fakeContext(`FROM scratch
	ADD ["file*.txt", "test"]
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using ADD with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildCopyMultipleFilesToFile(c *check.C) {
	name := "testcopymultiplefilestofile"
	ctx, err := fakeContext(`FROM scratch
	COPY file1.txt file2.txt test
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using COPY with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildJSONCopyMultipleFilesToFile(c *check.C) {
	name := "testjsoncopymultiplefilestofile"
	ctx, err := fakeContext(`FROM scratch
	COPY ["file1.txt", "file2.txt", "test"]
	`,
		map[string]string{
			"file1.txt": "test1",
			"file2.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using COPY with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildAddFileWithWhitespace(c *check.C) {
	name := "testaddfilewithwhitespace"
	ctx, err := fakeContext(`FROM busybox
RUN mkdir "/test dir"
RUN mkdir "/test_dir"
ADD [ "test file1", "/test_file1" ]
ADD [ "test_file2", "/test file2" ]
ADD [ "test file3", "/test file3" ]
ADD [ "test dir/test_file4", "/test_dir/test_file4" ]
ADD [ "test_dir/test_file5", "/test dir/test_file5" ]
ADD [ "test dir/test_file6", "/test dir/test_file6" ]
RUN [ $(cat "/test_file1") = 'test1' ]
RUN [ $(cat "/test file2") = 'test2' ]
RUN [ $(cat "/test file3") = 'test3' ]
RUN [ $(cat "/test_dir/test_file4") = 'test4' ]
RUN [ $(cat "/test dir/test_file5") = 'test5' ]
RUN [ $(cat "/test dir/test_file6") = 'test6' ]`,
		map[string]string{
			"test file1":          "test1",
			"test_file2":          "test2",
			"test file3":          "test3",
			"test dir/test_file4": "test4",
			"test_dir/test_file5": "test5",
			"test dir/test_file6": "test6",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyFileWithWhitespace(c *check.C) {
	name := "testcopyfilewithwhitespace"
	ctx, err := fakeContext(`FROM busybox
RUN mkdir "/test dir"
RUN mkdir "/test_dir"
COPY [ "test file1", "/test_file1" ]
COPY [ "test_file2", "/test file2" ]
COPY [ "test file3", "/test file3" ]
COPY [ "test dir/test_file4", "/test_dir/test_file4" ]
COPY [ "test_dir/test_file5", "/test dir/test_file5" ]
COPY [ "test dir/test_file6", "/test dir/test_file6" ]
RUN [ $(cat "/test_file1") = 'test1' ]
RUN [ $(cat "/test file2") = 'test2' ]
RUN [ $(cat "/test file3") = 'test3' ]
RUN [ $(cat "/test_dir/test_file4") = 'test4' ]
RUN [ $(cat "/test dir/test_file5") = 'test5' ]
RUN [ $(cat "/test dir/test_file6") = 'test6' ]`,
		map[string]string{
			"test file1":          "test1",
			"test_file2":          "test2",
			"test file3":          "test3",
			"test dir/test_file4": "test4",
			"test_dir/test_file5": "test5",
			"test dir/test_file6": "test6",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddMultipleFilesToFileWithWhitespace(c *check.C) {
	name := "testaddmultiplefilestofilewithwhitespace"
	ctx, err := fakeContext(`FROM busybox
	ADD [ "test file1", "test file2", "test" ]
    `,
		map[string]string{
			"test file1": "test1",
			"test file2": "test2",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using ADD with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildCopyMultipleFilesToFileWithWhitespace(c *check.C) {
	name := "testcopymultiplefilestofilewithwhitespace"
	ctx, err := fakeContext(`FROM busybox
	COPY [ "test file1", "test file2", "test" ]
        `,
		map[string]string{
			"test file1": "test1",
			"test file2": "test2",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "When using COPY with more than one source file, the destination must be a directory and end with a /"
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain %q) got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildCopyWildcard(c *check.C) {
	name := "testcopywildcard"
	server, err := fakeStorage(map[string]string{
		"robots.txt": "hello",
		"index.html": "world",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
	COPY file*.txt /tmp/
	RUN ls /tmp/file1.txt /tmp/file2.txt
	RUN mkdir /tmp1
	COPY dir* /tmp1/
	RUN ls /tmp1/dirt /tmp1/nested_file /tmp1/nested_dir/nest_nest_file
	RUN mkdir /tmp2
        ADD dir/*dir %s/robots.txt /tmp2/
	RUN ls /tmp2/nest_nest_file /tmp2/robots.txt
	`, server.URL()),
		map[string]string{
			"file1.txt":                     "test1",
			"file2.txt":                     "test2",
			"dir/nested_file":               "nested file",
			"dir/nested_dir/nest_nest_file": "2 times nested",
			"dirt": "dirty",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}

	// Now make sure we use a cache the 2nd time
	id2, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}

	if id1 != id2 {
		c.Fatal("didn't use the cache")
	}

}

func (s *DockerSuite) TestBuildCopyWildcardNoFind(c *check.C) {
	name := "testcopywildcardnofind"
	ctx, err := fakeContext(`FROM busybox
	COPY file*.txt /tmp/
	`, nil)
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	_, err = buildImageFromContext(name, ctx, true)
	if err == nil {
		c.Fatal("should have failed to find a file")
	}
	if !strings.Contains(err.Error(), "No source files were specified") {
		c.Fatalf("Wrong error %v, must be about no source files", err)
	}

}

func (s *DockerSuite) TestBuildCopyWildcardInName(c *check.C) {
	name := "testcopywildcardinname"
	ctx, err := fakeContext(`FROM busybox
	COPY *.txt /tmp/
	RUN [ "$(cat /tmp/\*.txt)" = 'hi there' ]
	`, map[string]string{"*.txt": "hi there"})

	if err != nil {
		// Normally we would do c.Fatal(err) here but given that
		// the odds of this failing are so rare, it must be because
		// the OS we're running the client on doesn't support * in
		// filenames (like windows).  So, instead of failing the test
		// just let it pass. Then we don't need to explicitly
		// say which OSs this works on or not.
		return
	}
	defer ctx.Close()

	_, err = buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatalf("should have built: %q", err)
	}
}

func (s *DockerSuite) TestBuildCopyWildcardCache(c *check.C) {
	name := "testcopywildcardcache"
	ctx, err := fakeContext(`FROM busybox
	COPY file1.txt /tmp/`,
		map[string]string{
			"file1.txt": "test1",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}

	// Now make sure we use a cache the 2nd time even with wild cards.
	// Use the same context so the file is the same and the checksum will match
	ctx.Add("Dockerfile", `FROM busybox
	COPY file*.txt /tmp/`)

	id2, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}

	if id1 != id2 {
		c.Fatal("didn't use the cache")
	}

}

func (s *DockerSuite) TestBuildAddSingleFileToNonExistingDir(c *check.C) {
	name := "testaddsinglefiletononexistingdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio /exists
ADD test_file /test_dir/
RUN [ $(ls -l / | grep test_dir | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildAddDirContentToRoot(c *check.C) {
	name := "testadddircontenttoroot"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio exists
ADD test_dir /
RUN [ $(ls -l /test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddDirContentToExistingDir(c *check.C) {
	name := "testadddircontenttoexistingdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN mkdir /exists
RUN touch /exists/exists_file
RUN chown -R dockerio.dockerio /exists
ADD test_dir/ /exists/
RUN [ $(ls -l / | grep exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/exists_file | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/test_file | awk '{print $3":"$4}') = 'root:root' ]`,
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddWholeDirToRoot(c *check.C) {
	name := "testaddwholedirtoroot"
	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio exists
ADD test_dir /test_dir
RUN [ $(ls -l / | grep test_dir | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l / | grep test_dir | awk '{print $1}') = 'drwxr-xr-x' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $1}') = '%s' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`, expectedFileChmod),
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

// Testing #5941
func (s *DockerSuite) TestBuildAddEtcToRoot(c *check.C) {
	name := "testaddetctoroot"
	ctx, err := fakeContext(`FROM scratch
ADD . /`,
		map[string]string{
			"etc/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

// Testing #9401
func (s *DockerSuite) TestBuildAddPreservesFilesSpecialBits(c *check.C) {
	name := "testaddpreservesfilesspecialbits"
	ctx, err := fakeContext(`FROM busybox
ADD suidbin /usr/bin/suidbin
RUN chmod 4755 /usr/bin/suidbin
RUN [ $(ls -l /usr/bin/suidbin | awk '{print $1}') = '-rwsr-xr-x' ]
ADD ./data/ /
RUN [ $(ls -l /usr/bin/suidbin | awk '{print $1}') = '-rwsr-xr-x' ]`,
		map[string]string{
			"suidbin":             "suidbin",
			"/data/usr/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopySingleFileToRoot(c *check.C) {
	name := "testcopysinglefiletoroot"
	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio /exists
COPY test_file /
RUN [ $(ls -l /test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_file | awk '{print $1}') = '%s' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`, expectedFileChmod),
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

// Issue #3960: "ADD src ." hangs - adapted for COPY
func (s *DockerSuite) TestBuildCopySingleFileToWorkdir(c *check.C) {
	name := "testcopysinglefiletoworkdir"
	ctx, err := fakeContext(`FROM busybox
COPY test_file .`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	errChan := make(chan error)
	go func() {
		_, err := buildImageFromContext(name, ctx, true)
		errChan <- err
		close(errChan)
	}()
	select {
	case <-time.After(5 * time.Second):
		c.Fatal("Build with adding to workdir timed out")
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	}
}

func (s *DockerSuite) TestBuildCopySingleFileToExistDir(c *check.C) {
	name := "testcopysinglefiletoexistdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN mkdir /exists
RUN touch /exists/exists_file
RUN chown -R dockerio.dockerio /exists
COPY test_file /exists/
RUN [ $(ls -l / | grep exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists/exists_file | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopySingleFileToNonExistDir(c *check.C) {
	name := "testcopysinglefiletononexistdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio /exists
COPY test_file /test_dir/
RUN [ $(ls -l / | grep test_dir | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyDirContentToRoot(c *check.C) {
	name := "testcopydircontenttoroot"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio exists
COPY test_dir /
RUN [ $(ls -l /test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`,
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyDirContentToExistDir(c *check.C) {
	name := "testcopydircontenttoexistdir"
	ctx, err := fakeContext(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN mkdir /exists
RUN touch /exists/exists_file
RUN chown -R dockerio.dockerio /exists
COPY test_dir/ /exists/
RUN [ $(ls -l / | grep exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/exists_file | awk '{print $3":"$4}') = 'dockerio:dockerio' ]
RUN [ $(ls -l /exists/test_file | awk '{print $3":"$4}') = 'root:root' ]`,
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyWholeDirToRoot(c *check.C) {
	name := "testcopywholedirtoroot"
	ctx, err := fakeContext(fmt.Sprintf(`FROM busybox
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group
RUN touch /exists
RUN chown dockerio.dockerio exists
COPY test_dir /test_dir
RUN [ $(ls -l / | grep test_dir | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l / | grep test_dir | awk '{print $1}') = 'drwxr-xr-x' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $3":"$4}') = 'root:root' ]
RUN [ $(ls -l /test_dir/test_file | awk '{print $1}') = '%s' ]
RUN [ $(ls -l /exists | awk '{print $3":"$4}') = 'dockerio:dockerio' ]`, expectedFileChmod),
		map[string]string{
			"test_dir/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyEtcToRoot(c *check.C) {
	name := "testcopyetctoroot"
	ctx, err := fakeContext(`FROM scratch
COPY . /`,
		map[string]string{
			"etc/test_file": "test1",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCopyDisallowRemote(c *check.C) {
	name := "testcopydisallowremote"
	_, out, err := buildImageWithOut(name, `FROM scratch
COPY https://index.docker.io/robots.txt /`,
		true)
	if err == nil || !strings.Contains(out, "Source can't be a URL for COPY") {
		c.Fatalf("Error should be about disallowed remote source, got err: %s, out: %q", err, out)
	}
}

func (s *DockerSuite) TestBuildAddBadLinks(c *check.C) {
	const (
		dockerfile = `
			FROM scratch
			ADD links.tar /
			ADD foo.txt /symlink/
			`
		targetFile = "foo.txt"
	)
	var (
		name = "test-link-absolute"
	)
	ctx, err := fakeContext(dockerfile, nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	tempDir, err := ioutil.TempDir("", "test-link-absolute-temp-")
	if err != nil {
		c.Fatalf("failed to create temporary directory: %s", tempDir)
	}
	defer os.RemoveAll(tempDir)

	var symlinkTarget string
	if runtime.GOOS == "windows" {
		var driveLetter string
		if abs, err := filepath.Abs(tempDir); err != nil {
			c.Fatal(err)
		} else {
			driveLetter = abs[:1]
		}
		tempDirWithoutDrive := tempDir[2:]
		symlinkTarget = fmt.Sprintf(`%s:\..\..\..\..\..\..\..\..\..\..\..\..%s`, driveLetter, tempDirWithoutDrive)
	} else {
		symlinkTarget = fmt.Sprintf("/../../../../../../../../../../../..%s", tempDir)
	}

	tarPath := filepath.Join(ctx.Dir, "links.tar")
	nonExistingFile := filepath.Join(tempDir, targetFile)
	fooPath := filepath.Join(ctx.Dir, targetFile)

	tarOut, err := os.Create(tarPath)
	if err != nil {
		c.Fatal(err)
	}

	tarWriter := tar.NewWriter(tarOut)

	header := &tar.Header{
		Name:     "symlink",
		Typeflag: tar.TypeSymlink,
		Linkname: symlinkTarget,
		Mode:     0755,
		Uid:      0,
		Gid:      0,
	}

	err = tarWriter.WriteHeader(header)
	if err != nil {
		c.Fatal(err)
	}

	tarWriter.Close()
	tarOut.Close()

	foo, err := os.Create(fooPath)
	if err != nil {
		c.Fatal(err)
	}
	defer foo.Close()

	if _, err := foo.WriteString("test"); err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

	if _, err := os.Stat(nonExistingFile); err == nil || err != nil && !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't have been written and it shouldn't exist", nonExistingFile)
	}

}

func (s *DockerSuite) TestBuildAddBadLinksVolume(c *check.C) {
	const (
		dockerfileTemplate = `
		FROM busybox
		RUN ln -s /../../../../../../../../%s /x
		VOLUME /x
		ADD foo.txt /x/`
		targetFile = "foo.txt"
	)
	var (
		name       = "test-link-absolute-volume"
		dockerfile = ""
	)

	tempDir, err := ioutil.TempDir("", "test-link-absolute-volume-temp-")
	if err != nil {
		c.Fatalf("failed to create temporary directory: %s", tempDir)
	}
	defer os.RemoveAll(tempDir)

	dockerfile = fmt.Sprintf(dockerfileTemplate, tempDir)
	nonExistingFile := filepath.Join(tempDir, targetFile)

	ctx, err := fakeContext(dockerfile, nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	fooPath := filepath.Join(ctx.Dir, targetFile)

	foo, err := os.Create(fooPath)
	if err != nil {
		c.Fatal(err)
	}
	defer foo.Close()

	if _, err := foo.WriteString("test"); err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

	if _, err := os.Stat(nonExistingFile); err == nil || err != nil && !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't have been written and it shouldn't exist", nonExistingFile)
	}

}

// Issue #5270 - ensure we throw a better error than "unexpected EOF"
// when we can't access files in the context.
func (s *DockerSuite) TestBuildWithInaccessibleFilesInContext(c *check.C) {
	testRequires(c, UnixCli) // test uses chown/chmod: not available on windows

	{
		name := "testbuildinaccessiblefiles"
		ctx, err := fakeContext("FROM scratch\nADD . /foo/", map[string]string{"fileWithoutReadAccess": "foo"})
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()
		// This is used to ensure we detect inaccessible files early during build in the cli client
		pathToFileWithoutReadAccess := filepath.Join(ctx.Dir, "fileWithoutReadAccess")

		if err = os.Chown(pathToFileWithoutReadAccess, 0, 0); err != nil {
			c.Fatalf("failed to chown file to root: %s", err)
		}
		if err = os.Chmod(pathToFileWithoutReadAccess, 0700); err != nil {
			c.Fatalf("failed to chmod file to 700: %s", err)
		}
		buildCmd := exec.Command("su", "unprivilegeduser", "-c", fmt.Sprintf("%s build -t %s .", dockerBinary, name))
		buildCmd.Dir = ctx.Dir
		out, _, err := runCommandWithOutput(buildCmd)
		if err == nil {
			c.Fatalf("build should have failed: %s %s", err, out)
		}

		// check if we've detected the failure before we started building
		if !strings.Contains(out, "no permission to read from ") {
			c.Fatalf("output should've contained the string: no permission to read from but contained: %s", out)
		}

		if !strings.Contains(out, "Error checking context") {
			c.Fatalf("output should've contained the string: Error checking context")
		}
	}
	{
		name := "testbuildinaccessibledirectory"
		ctx, err := fakeContext("FROM scratch\nADD . /foo/", map[string]string{"directoryWeCantStat/bar": "foo"})
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()
		// This is used to ensure we detect inaccessible directories early during build in the cli client
		pathToDirectoryWithoutReadAccess := filepath.Join(ctx.Dir, "directoryWeCantStat")
		pathToFileInDirectoryWithoutReadAccess := filepath.Join(pathToDirectoryWithoutReadAccess, "bar")

		if err = os.Chown(pathToDirectoryWithoutReadAccess, 0, 0); err != nil {
			c.Fatalf("failed to chown directory to root: %s", err)
		}
		if err = os.Chmod(pathToDirectoryWithoutReadAccess, 0444); err != nil {
			c.Fatalf("failed to chmod directory to 444: %s", err)
		}
		if err = os.Chmod(pathToFileInDirectoryWithoutReadAccess, 0700); err != nil {
			c.Fatalf("failed to chmod file to 700: %s", err)
		}

		buildCmd := exec.Command("su", "unprivilegeduser", "-c", fmt.Sprintf("%s build -t %s .", dockerBinary, name))
		buildCmd.Dir = ctx.Dir
		out, _, err := runCommandWithOutput(buildCmd)
		if err == nil {
			c.Fatalf("build should have failed: %s %s", err, out)
		}

		// check if we've detected the failure before we started building
		if !strings.Contains(out, "can't stat") {
			c.Fatalf("output should've contained the string: can't access %s", out)
		}

		if !strings.Contains(out, "Error checking context") {
			c.Fatalf("output should've contained the string: Error checking context\ngot:%s", out)
		}

	}
	{
		name := "testlinksok"
		ctx, err := fakeContext("FROM scratch\nADD . /foo/", nil)
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()

		target := "../../../../../../../../../../../../../../../../../../../azA"
		if err := os.Symlink(filepath.Join(ctx.Dir, "g"), target); err != nil {
			c.Fatal(err)
		}
		defer os.Remove(target)
		// This is used to ensure we don't follow links when checking if everything in the context is accessible
		// This test doesn't require that we run commands as an unprivileged user
		if _, err := buildImageFromContext(name, ctx, true); err != nil {
			c.Fatal(err)
		}
	}
	{
		name := "testbuildignoredinaccessible"
		ctx, err := fakeContext("FROM scratch\nADD . /foo/",
			map[string]string{
				"directoryWeCantStat/bar": "foo",
				".dockerignore":           "directoryWeCantStat",
			})
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()
		// This is used to ensure we don't try to add inaccessible files when they are ignored by a .dockerignore pattern
		pathToDirectoryWithoutReadAccess := filepath.Join(ctx.Dir, "directoryWeCantStat")
		pathToFileInDirectoryWithoutReadAccess := filepath.Join(pathToDirectoryWithoutReadAccess, "bar")
		if err = os.Chown(pathToDirectoryWithoutReadAccess, 0, 0); err != nil {
			c.Fatalf("failed to chown directory to root: %s", err)
		}
		if err = os.Chmod(pathToDirectoryWithoutReadAccess, 0444); err != nil {
			c.Fatalf("failed to chmod directory to 755: %s", err)
		}
		if err = os.Chmod(pathToFileInDirectoryWithoutReadAccess, 0700); err != nil {
			c.Fatalf("failed to chmod file to 444: %s", err)
		}

		buildCmd := exec.Command("su", "unprivilegeduser", "-c", fmt.Sprintf("%s build -t %s .", dockerBinary, name))
		buildCmd.Dir = ctx.Dir
		if out, _, err := runCommandWithOutput(buildCmd); err != nil {
			c.Fatalf("build should have worked: %s %s", err, out)
		}

	}
}

func (s *DockerSuite) TestBuildForceRm(c *check.C) {
	containerCountBefore, err := getContainerCount()
	if err != nil {
		c.Fatalf("failed to get the container count: %s", err)
	}
	name := "testbuildforcerm"
	ctx, err := fakeContext("FROM scratch\nRUN true\nRUN thiswillfail", nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	dockerCmdInDir(c, ctx.Dir, "build", "-t", name, "--force-rm", ".")

	containerCountAfter, err := getContainerCount()
	if err != nil {
		c.Fatalf("failed to get the container count: %s", err)
	}

	if containerCountBefore != containerCountAfter {
		c.Fatalf("--force-rm shouldn't have left containers behind")
	}

}

// Test that an infinite sleep during a build is killed if the client disconnects.
// This test is fairly hairy because there are lots of ways to race.
// Strategy:
// * Monitor the output of docker events starting from before
// * Run a 1-year-long sleep from a docker build.
// * When docker events sees container start, close the "docker build" command
// * Wait for docker events to emit a dying event.
func (s *DockerSuite) TestBuildCancelationKillsSleep(c *check.C) {
	name := "testbuildcancelation"

	// (Note: one year, will never finish)
	ctx, err := fakeContext("FROM busybox\nRUN sleep 31536000", nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	finish := make(chan struct{})
	defer close(finish)

	eventStart := make(chan struct{})
	eventDie := make(chan struct{})
	containerID := make(chan string)

	startEpoch := daemonTime(c).Unix()
	// Watch for events since epoch.
	eventsCmd := exec.Command(
		dockerBinary, "events",
		"--since", strconv.FormatInt(startEpoch, 10))
	stdout, err := eventsCmd.StdoutPipe()
	if err != nil {
		c.Fatal(err)
	}
	if err := eventsCmd.Start(); err != nil {
		c.Fatal(err)
	}
	defer eventsCmd.Process.Kill()

	// Goroutine responsible for watching start/die events from `docker events`
	go func() {
		cid := <-containerID

		matchStart := regexp.MustCompile(cid + `(.*) start$`)
		matchDie := regexp.MustCompile(cid + `(.*) die$`)

		//
		// Read lines of `docker events` looking for container start and stop.
		//
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			switch {
			case matchStart.MatchString(scanner.Text()):
				close(eventStart)
			case matchDie.MatchString(scanner.Text()):
				close(eventDie)
			}
		}
	}()

	buildCmd := exec.Command(dockerBinary, "build", "-t", name, ".")
	buildCmd.Dir = ctx.Dir

	stdoutBuild, err := buildCmd.StdoutPipe()
	if err := buildCmd.Start(); err != nil {
		c.Fatalf("failed to run build: %s", err)
	}

	matchCID := regexp.MustCompile("Running in ")
	scanner := bufio.NewScanner(stdoutBuild)
	for scanner.Scan() {
		line := scanner.Text()
		if ok := matchCID.MatchString(line); ok {
			containerID <- line[len(line)-12:]
			break
		}
	}

	select {
	case <-time.After(5 * time.Second):
		c.Fatal("failed to observe build container start in timely fashion")
	case <-eventStart:
		// Proceeds from here when we see the container fly past in the
		// output of "docker events".
		// Now we know the container is running.
	}

	// Send a kill to the `docker build` command.
	// Causes the underlying build to be cancelled due to socket close.
	if err := buildCmd.Process.Kill(); err != nil {
		c.Fatalf("error killing build command: %s", err)
	}

	// Get the exit status of `docker build`, check it exited because killed.
	if err := buildCmd.Wait(); err != nil && !IsKilled(err) {
		c.Fatalf("wait failed during build run: %T %s", err, err)
	}

	select {
	case <-time.After(5 * time.Second):
		// If we don't get here in a timely fashion, it wasn't killed.
		c.Fatal("container cancel did not succeed")
	case <-eventDie:
		// We saw the container shut down in the `docker events` stream,
		// as expected.
	}

}

func (s *DockerSuite) TestBuildRm(c *check.C) {
	name := "testbuildrm"
	ctx, err := fakeContext("FROM scratch\nADD foo /\nADD foo /", map[string]string{"foo": "bar"})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	{
		containerCountBefore, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "--rm", "-t", name, ".")

		if err != nil {
			c.Fatal("failed to build the image", out)
		}

		containerCountAfter, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		if containerCountBefore != containerCountAfter {
			c.Fatalf("-rm shouldn't have left containers behind")
		}
		deleteImages(name)
	}

	{
		containerCountBefore, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-t", name, ".")

		if err != nil {
			c.Fatal("failed to build the image", out)
		}

		containerCountAfter, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		if containerCountBefore != containerCountAfter {
			c.Fatalf("--rm shouldn't have left containers behind")
		}
		deleteImages(name)
	}

	{
		containerCountBefore, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "--rm=false", "-t", name, ".")

		if err != nil {
			c.Fatal("failed to build the image", out)
		}

		containerCountAfter, err := getContainerCount()
		if err != nil {
			c.Fatalf("failed to get the container count: %s", err)
		}

		if containerCountBefore == containerCountAfter {
			c.Fatalf("--rm=false should have left containers behind")
		}
		deleteImages(name)

	}

}

func (s *DockerSuite) TestBuildWithVolumes(c *check.C) {
	var (
		result   map[string]map[string]struct{}
		name     = "testbuildvolumes"
		emptyMap = make(map[string]struct{})
		expected = map[string]map[string]struct{}{
			"/test1":  emptyMap,
			"/test2":  emptyMap,
			"/test3":  emptyMap,
			"/test4":  emptyMap,
			"/test5":  emptyMap,
			"/test6":  emptyMap,
			"[/test7": emptyMap,
			"/test8]": emptyMap,
		}
	)
	_, err := buildImage(name,
		`FROM scratch
		VOLUME /test1
		VOLUME /test2
    VOLUME /test3 /test4
    VOLUME ["/test5", "/test6"]
    VOLUME [/test7 /test8]
    `,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectFieldJSON(name, "Config.Volumes")
	if err != nil {
		c.Fatal(err)
	}

	err = unmarshalJSON([]byte(res), &result)
	if err != nil {
		c.Fatal(err)
	}

	equal := reflect.DeepEqual(&result, &expected)

	if !equal {
		c.Fatalf("Volumes %s, expected %s", result, expected)
	}

}

func (s *DockerSuite) TestBuildMaintainer(c *check.C) {
	name := "testbuildmaintainer"
	expected := "dockerio"
	_, err := buildImage(name,
		`FROM scratch
        MAINTAINER dockerio`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Author")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Maintainer %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildUser(c *check.C) {
	name := "testbuilduser"
	expected := "dockerio"
	_, err := buildImage(name,
		`FROM busybox
		RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
		USER dockerio
		RUN [ $(whoami) = 'dockerio' ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.User")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("User %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildRelativeWorkdir(c *check.C) {
	name := "testbuildrelativeworkdir"
	expected := "/test2/test3"
	_, err := buildImage(name,
		`FROM busybox
		RUN [ "$PWD" = '/' ]
		WORKDIR test1
		RUN [ "$PWD" = '/test1' ]
		WORKDIR /test2
		RUN [ "$PWD" = '/test2' ]
		WORKDIR test3
		RUN [ "$PWD" = '/test2/test3' ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.WorkingDir")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Workdir %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildWorkdirWithEnvVariables(c *check.C) {
	name := "testbuildworkdirwithenvvariables"
	expected := "/test1/test2"
	_, err := buildImage(name,
		`FROM busybox
		ENV DIRPATH /test1
		ENV SUBDIRNAME test2
		WORKDIR $DIRPATH
		WORKDIR $SUBDIRNAME/$MISSING_VAR`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.WorkingDir")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Workdir %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildRelativeCopy(c *check.C) {
	name := "testbuildrelativecopy"
	dockerfile := `
		FROM busybox
			WORKDIR /test1
			WORKDIR test2
			RUN [ "$PWD" = '/test1/test2' ]
			COPY foo ./
			RUN [ "$(cat /test1/test2/foo)" = 'hello' ]
			ADD foo ./bar/baz
			RUN [ "$(cat /test1/test2/bar/baz)" = 'hello' ]
			COPY foo ./bar/baz2
			RUN [ "$(cat /test1/test2/bar/baz2)" = 'hello' ]
			WORKDIR ..
			COPY foo ./
			RUN [ "$(cat /test1/foo)" = 'hello' ]
			COPY foo /test3/
			RUN [ "$(cat /test3/foo)" = 'hello' ]
			WORKDIR /test4
			COPY . .
			RUN [ "$(cat /test4/foo)" = 'hello' ]
			WORKDIR /test5/test6
			COPY foo ../
			RUN [ "$(cat /test5/foo)" = 'hello' ]
			`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	_, err = buildImageFromContext(name, ctx, false)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildEnv(c *check.C) {
	name := "testbuildenv"
	expected := "[PATH=/test:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PORT=2375]"
	_, err := buildImage(name,
		`FROM busybox
		ENV PATH /test:$PATH
        ENV PORT 2375
		RUN [ $(env | grep PORT) = 'PORT=2375' ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Env")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Env %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildContextCleanup(c *check.C) {
	testRequires(c, SameHostDaemon)

	name := "testbuildcontextcleanup"
	entries, err := ioutil.ReadDir("/var/lib/docker/tmp")
	if err != nil {
		c.Fatalf("failed to list contents of tmp dir: %s", err)
	}
	_, err = buildImage(name,
		`FROM scratch
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	entriesFinal, err := ioutil.ReadDir("/var/lib/docker/tmp")
	if err != nil {
		c.Fatalf("failed to list contents of tmp dir: %s", err)
	}
	if err = compareDirectoryEntries(entries, entriesFinal); err != nil {
		c.Fatalf("context should have been deleted, but wasn't")
	}

}

func (s *DockerSuite) TestBuildContextCleanupFailedBuild(c *check.C) {
	testRequires(c, SameHostDaemon)

	name := "testbuildcontextcleanup"
	entries, err := ioutil.ReadDir("/var/lib/docker/tmp")
	if err != nil {
		c.Fatalf("failed to list contents of tmp dir: %s", err)
	}
	_, err = buildImage(name,
		`FROM scratch
	RUN /non/existing/command`,
		true)
	if err == nil {
		c.Fatalf("expected build to fail, but it didn't")
	}
	entriesFinal, err := ioutil.ReadDir("/var/lib/docker/tmp")
	if err != nil {
		c.Fatalf("failed to list contents of tmp dir: %s", err)
	}
	if err = compareDirectoryEntries(entries, entriesFinal); err != nil {
		c.Fatalf("context should have been deleted, but wasn't")
	}

}

func (s *DockerSuite) TestBuildCmd(c *check.C) {
	name := "testbuildcmd"
	expected := "{[/bin/echo Hello World]}"
	_, err := buildImage(name,
		`FROM scratch
        CMD ["/bin/echo", "Hello World"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Cmd %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildExpose(c *check.C) {
	name := "testbuildexpose"
	expected := "map[2375/tcp:{}]"
	_, err := buildImage(name,
		`FROM scratch
        EXPOSE 2375`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Exposed ports %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildExposeMorePorts(c *check.C) {
	// start building docker file with a large number of ports
	portList := make([]string, 50)
	line := make([]string, 100)
	expectedPorts := make([]int, len(portList)*len(line))
	for i := 0; i < len(portList); i++ {
		for j := 0; j < len(line); j++ {
			p := i*len(line) + j + 1
			line[j] = strconv.Itoa(p)
			expectedPorts[p-1] = p
		}
		if i == len(portList)-1 {
			portList[i] = strings.Join(line, " ")
		} else {
			portList[i] = strings.Join(line, " ") + ` \`
		}
	}

	dockerfile := `FROM scratch
	EXPOSE {{range .}} {{.}}
	{{end}}`
	tmpl := template.Must(template.New("dockerfile").Parse(dockerfile))
	buf := bytes.NewBuffer(nil)
	tmpl.Execute(buf, portList)

	name := "testbuildexpose"
	_, err := buildImage(name, buf.String(), true)
	if err != nil {
		c.Fatal(err)
	}

	// check if all the ports are saved inside Config.ExposedPorts
	res, err := inspectFieldJSON(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}
	var exposedPorts map[string]interface{}
	if err := json.Unmarshal([]byte(res), &exposedPorts); err != nil {
		c.Fatal(err)
	}

	for _, p := range expectedPorts {
		ep := fmt.Sprintf("%d/tcp", p)
		if _, ok := exposedPorts[ep]; !ok {
			c.Errorf("Port(%s) is not exposed", ep)
		} else {
			delete(exposedPorts, ep)
		}
	}
	if len(exposedPorts) != 0 {
		c.Errorf("Unexpected extra exposed ports %v", exposedPorts)
	}
}

func (s *DockerSuite) TestBuildExposeOrder(c *check.C) {
	buildID := func(name, exposed string) string {
		_, err := buildImage(name, fmt.Sprintf(`FROM scratch
		EXPOSE %s`, exposed), true)
		if err != nil {
			c.Fatal(err)
		}
		id, err := inspectField(name, "Id")
		if err != nil {
			c.Fatal(err)
		}
		return id
	}

	id1 := buildID("testbuildexpose1", "80 2375")
	id2 := buildID("testbuildexpose2", "2375 80")
	if id1 != id2 {
		c.Errorf("EXPOSE should invalidate the cache only when ports actually changed")
	}
}

func (s *DockerSuite) TestBuildExposeUpperCaseProto(c *check.C) {
	name := "testbuildexposeuppercaseproto"
	expected := "map[5678/udp:{}]"
	_, err := buildImage(name,
		`FROM scratch
        EXPOSE 5678/UDP`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Exposed ports %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildEmptyEntrypointInheritance(c *check.C) {
	name := "testbuildentrypointinheritance"
	name2 := "testbuildentrypointinheritance2"

	_, err := buildImage(name,
		`FROM busybox
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}

	expected := "{[/bin/echo]}"
	if res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}

	_, err = buildImage(name2,
		fmt.Sprintf(`FROM %s
        ENTRYPOINT []`, name),
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err = inspectField(name2, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}

	expected = "{[]}"

	if res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}

}

func (s *DockerSuite) TestBuildEmptyEntrypoint(c *check.C) {
	name := "testbuildentrypoint"
	expected := "{[]}"

	_, err := buildImage(name,
		`FROM busybox
        ENTRYPOINT []`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}

}

func (s *DockerSuite) TestBuildEntrypoint(c *check.C) {
	name := "testbuildentrypoint"
	expected := "{[/bin/echo]}"
	_, err := buildImage(name,
		`FROM scratch
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}

}

// #6445 ensure ONBUILD triggers aren't committed to grandchildren
func (s *DockerSuite) TestBuildOnBuildLimitedInheritence(c *check.C) {
	var (
		out2, out3 string
	)
	{
		name1 := "testonbuildtrigger1"
		dockerfile1 := `
		FROM busybox
		RUN echo "GRANDPARENT"
		ONBUILD RUN echo "ONBUILD PARENT"
		`
		ctx, err := fakeContext(dockerfile1, nil)
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()

		out1, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-t", name1, ".")
		if err != nil {
			c.Fatalf("build failed to complete: %s, %v", out1, err)
		}
	}
	{
		name2 := "testonbuildtrigger2"
		dockerfile2 := `
		FROM testonbuildtrigger1
		`
		ctx, err := fakeContext(dockerfile2, nil)
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()

		out2, _, err = dockerCmdInDir(c, ctx.Dir, "build", "-t", name2, ".")
		if err != nil {
			c.Fatalf("build failed to complete: %s, %v", out2, err)
		}
	}
	{
		name3 := "testonbuildtrigger3"
		dockerfile3 := `
		FROM testonbuildtrigger2
		`
		ctx, err := fakeContext(dockerfile3, nil)
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()

		out3, _, err = dockerCmdInDir(c, ctx.Dir, "build", "-t", name3, ".")
		if err != nil {
			c.Fatalf("build failed to complete: %s, %v", out3, err)
		}

	}

	// ONBUILD should be run in second build.
	if !strings.Contains(out2, "ONBUILD PARENT") {
		c.Fatalf("ONBUILD instruction did not run in child of ONBUILD parent")
	}

	// ONBUILD should *not* be run in third build.
	if strings.Contains(out3, "ONBUILD PARENT") {
		c.Fatalf("ONBUILD instruction ran in grandchild of ONBUILD parent")
	}

}

func (s *DockerSuite) TestBuildWithCache(c *check.C) {
	name := "testbuildwithcache"
	id1, err := buildImage(name,
		`FROM scratch
		MAINTAINER dockerio
		EXPOSE 5432
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImage(name,
		`FROM scratch
		MAINTAINER dockerio
		EXPOSE 5432
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func (s *DockerSuite) TestBuildWithoutCache(c *check.C) {
	name := "testbuildwithoutcache"
	name2 := "testbuildwithoutcache2"
	id1, err := buildImage(name,
		`FROM scratch
		MAINTAINER dockerio
		EXPOSE 5432
        ENTRYPOINT ["/bin/echo"]`,
		true)
	if err != nil {
		c.Fatal(err)
	}

	id2, err := buildImage(name2,
		`FROM scratch
		MAINTAINER dockerio
		EXPOSE 5432
        ENTRYPOINT ["/bin/echo"]`,
		false)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
}

func (s *DockerSuite) TestBuildConditionalCache(c *check.C) {
	name := "testbuildconditionalcache"

	dockerfile := `
		FROM busybox
        ADD foo /tmp/`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatalf("Error building #1: %s", err)
	}

	if err := ctx.Add("foo", "bye"); err != nil {
		c.Fatalf("Error modifying foo: %s", err)
	}

	id2, err := buildImageFromContext(name, ctx, false)
	if err != nil {
		c.Fatalf("Error building #2: %s", err)
	}
	if id2 == id1 {
		c.Fatal("Should not have used the cache")
	}

	id3, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatalf("Error building #3: %s", err)
	}
	if id3 != id2 {
		c.Fatal("Should have used the cache")
	}
}

func (s *DockerSuite) TestBuildAddLocalFileWithCache(c *check.C) {
	name := "testbuildaddlocalfilewithcache"
	name2 := "testbuildaddlocalfilewithcache2"
	dockerfile := `
		FROM busybox
        MAINTAINER dockerio
        ADD foo /usr/lib/bla/bar
		RUN [ "$(cat /usr/lib/bla/bar)" = "hello" ]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddMultipleLocalFileWithCache(c *check.C) {
	name := "testbuildaddmultiplelocalfilewithcache"
	name2 := "testbuildaddmultiplelocalfilewithcache2"
	dockerfile := `
		FROM busybox
        MAINTAINER dockerio
        ADD foo Dockerfile /usr/lib/bla/
		RUN [ "$(cat /usr/lib/bla/foo)" = "hello" ]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddLocalFileWithoutCache(c *check.C) {
	name := "testbuildaddlocalfilewithoutcache"
	name2 := "testbuildaddlocalfilewithoutcache2"
	dockerfile := `
		FROM busybox
        MAINTAINER dockerio
        ADD foo /usr/lib/bla/bar
		RUN [ "$(cat /usr/lib/bla/bar)" = "hello" ]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, false)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
}

func (s *DockerSuite) TestBuildCopyDirButNotFile(c *check.C) {
	name := "testbuildcopydirbutnotfile"
	name2 := "testbuildcopydirbutnotfile2"
	dockerfile := `
        FROM scratch
        COPY dir /tmp/`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"dir/foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	// Check that adding file with similar name doesn't mess with cache
	if err := ctx.Add("dir_file", "hello2"); err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but wasn't")
	}
}

func (s *DockerSuite) TestBuildAddCurrentDirWithCache(c *check.C) {
	name := "testbuildaddcurrentdirwithcache"
	name2 := name + "2"
	name3 := name + "3"
	name4 := name + "4"
	dockerfile := `
        FROM scratch
        MAINTAINER dockerio
        ADD . /usr/lib/bla`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	// Check that adding file invalidate cache of "ADD ."
	if err := ctx.Add("bar", "hello2"); err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
	// Check that changing file invalidate cache of "ADD ."
	if err := ctx.Add("foo", "hello1"); err != nil {
		c.Fatal(err)
	}
	id3, err := buildImageFromContext(name3, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id2 == id3 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
	// Check that changing file to same content with different mtime does not
	// invalidate cache of "ADD ."
	time.Sleep(1 * time.Second) // wait second because of mtime precision
	if err := ctx.Add("foo", "hello1"); err != nil {
		c.Fatal(err)
	}
	id4, err := buildImageFromContext(name4, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id3 != id4 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddCurrentDirWithoutCache(c *check.C) {
	name := "testbuildaddcurrentdirwithoutcache"
	name2 := "testbuildaddcurrentdirwithoutcache2"
	dockerfile := `
        FROM scratch
        MAINTAINER dockerio
        ADD . /usr/lib/bla`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo": "hello",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, false)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddRemoteFileWithCache(c *check.C) {
	name := "testbuildaddremotefilewithcache"
	server, err := fakeStorage(map[string]string{
		"baz": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	id1, err := buildImage(name,
		fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImage(name,
		fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddRemoteFileWithoutCache(c *check.C) {
	name := "testbuildaddremotefilewithoutcache"
	name2 := "testbuildaddremotefilewithoutcache2"
	server, err := fakeStorage(map[string]string{
		"baz": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	id1, err := buildImage(name,
		fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImage(name2,
		fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		false)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
}

func (s *DockerSuite) TestBuildAddRemoteFileMTime(c *check.C) {
	name := "testbuildaddremotefilemtime"
	name2 := name + "2"
	name3 := name + "3"

	files := map[string]string{"baz": "hello"}
	server, err := fakeStorage(files)
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	ctx, err := fakeContext(fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server.URL()), nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}

	id2, err := buildImageFromContext(name2, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but wasn't - #1")
	}

	// Now create a different server with same contents (causes different mtime)
	// The cache should still be used

	// allow some time for clock to pass as mtime precision is only 1s
	time.Sleep(2 * time.Second)

	server2, err := fakeStorage(files)
	if err != nil {
		c.Fatal(err)
	}
	defer server2.Close()

	ctx2, err := fakeContext(fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD %s/baz /usr/lib/baz/quux`, server2.URL()), nil)
	if err != nil {
		c.Fatal(err)
	}
	defer ctx2.Close()
	id3, err := buildImageFromContext(name3, ctx2, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id3 {
		c.Fatal("The cache should have been used but wasn't")
	}
}

func (s *DockerSuite) TestBuildAddLocalAndRemoteFilesWithCache(c *check.C) {
	name := "testbuildaddlocalandremotefilewithcache"
	server, err := fakeStorage(map[string]string{
		"baz": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	ctx, err := fakeContext(fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD foo /usr/lib/bla/bar
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		map[string]string{
			"foo": "hello world",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	if id1 != id2 {
		c.Fatal("The cache should have been used but hasn't.")
	}
}

func testContextTar(c *check.C, compression archive.Compression) {
	ctx, err := fakeContext(
		`FROM busybox
ADD foo /foo
CMD ["cat", "/foo"]`,
		map[string]string{
			"foo": "bar",
		},
	)
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	context, err := archive.Tar(ctx.Dir, compression)
	if err != nil {
		c.Fatalf("failed to build context tar: %v", err)
	}
	name := "contexttar"
	buildCmd := exec.Command(dockerBinary, "build", "-t", name, "-")
	buildCmd.Stdin = context

	if out, _, err := runCommandWithOutput(buildCmd); err != nil {
		c.Fatalf("build failed to complete: %v %v", out, err)
	}
}

func (s *DockerSuite) TestBuildContextTarGzip(c *check.C) {
	testContextTar(c, archive.Gzip)
}

func (s *DockerSuite) TestBuildContextTarNoCompression(c *check.C) {
	testContextTar(c, archive.Uncompressed)
}

func (s *DockerSuite) TestBuildNoContext(c *check.C) {
	buildCmd := exec.Command(dockerBinary, "build", "-t", "nocontext", "-")
	buildCmd.Stdin = strings.NewReader("FROM busybox\nCMD echo ok\n")

	if out, _, err := runCommandWithOutput(buildCmd); err != nil {
		c.Fatalf("build failed to complete: %v %v", out, err)
	}

	if out, _ := dockerCmd(c, "run", "--rm", "nocontext"); out != "ok\n" {
		c.Fatalf("run produced invalid output: %q, expected %q", out, "ok")
	}
}

// TODO: TestCaching
func (s *DockerSuite) TestBuildAddLocalAndRemoteFilesWithoutCache(c *check.C) {
	name := "testbuildaddlocalandremotefilewithoutcache"
	name2 := "testbuildaddlocalandremotefilewithoutcache2"
	server, err := fakeStorage(map[string]string{
		"baz": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	ctx, err := fakeContext(fmt.Sprintf(`FROM scratch
        MAINTAINER dockerio
        ADD foo /usr/lib/bla/bar
        ADD %s/baz /usr/lib/baz/quux`, server.URL()),
		map[string]string{
			"foo": "hello world",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	id1, err := buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
	id2, err := buildImageFromContext(name2, ctx, false)
	if err != nil {
		c.Fatal(err)
	}
	if id1 == id2 {
		c.Fatal("The cache should have been invalided but hasn't.")
	}
}

func (s *DockerSuite) TestBuildWithVolumeOwnership(c *check.C) {
	name := "testbuildimg"

	_, err := buildImage(name,
		`FROM busybox:latest
        RUN mkdir /test && chown daemon:daemon /test && chmod 0600 /test
        VOLUME /test`,
		true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "--rm", "testbuildimg", "ls", "-la", "/test")

	if expected := "drw-------"; !strings.Contains(out, expected) {
		c.Fatalf("expected %s received %s", expected, out)
	}

	if expected := "daemon   daemon"; !strings.Contains(out, expected) {
		c.Fatalf("expected %s received %s", expected, out)
	}

}

// testing #1405 - config.Cmd does not get cleaned up if
// utilizing cache
func (s *DockerSuite) TestBuildEntrypointRunCleanup(c *check.C) {
	name := "testbuildcmdcleanup"
	if _, err := buildImage(name,
		`FROM busybox
        RUN echo "hello"`,
		true); err != nil {
		c.Fatal(err)
	}

	ctx, err := fakeContext(`FROM busybox
        RUN echo "hello"
        ADD foo /foo
        ENTRYPOINT ["/bin/echo"]`,
		map[string]string{
			"foo": "hello",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	// Cmd must be cleaned up
	if res != "<nil>" {
		c.Fatalf("Cmd %s, expected nil", res)
	}
}

func (s *DockerSuite) TestBuildForbiddenContextPath(c *check.C) {
	name := "testbuildforbidpath"
	ctx, err := fakeContext(`FROM scratch
        ADD ../../ test/
        `,
		map[string]string{
			"test.txt":  "test1",
			"other.txt": "other",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	expected := "Forbidden path outside the build context: ../../ "
	if _, err := buildImageFromContext(name, ctx, true); err == nil || !strings.Contains(err.Error(), expected) {
		c.Fatalf("Wrong error: (should contain \"%s\") got:\n%v", expected, err)
	}

}

func (s *DockerSuite) TestBuildAddFileNotFound(c *check.C) {
	name := "testbuildaddnotfound"
	ctx, err := fakeContext(`FROM scratch
        ADD foo /usr/local/bar`,
		map[string]string{"bar": "hello"})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		if !strings.Contains(err.Error(), "foo: no such file or directory") {
			c.Fatalf("Wrong error %v, must be about missing foo file or directory", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

func (s *DockerSuite) TestBuildInheritance(c *check.C) {
	name := "testbuildinheritance"

	_, err := buildImage(name,
		`FROM scratch
		EXPOSE 2375`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	ports1, err := inspectField(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}

	_, err = buildImage(name,
		fmt.Sprintf(`FROM %s
		ENTRYPOINT ["/bin/echo"]`, name),
		true)
	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectField(name, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}
	if expected := "{[/bin/echo]}"; res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}
	ports2, err := inspectField(name, "Config.ExposedPorts")
	if err != nil {
		c.Fatal(err)
	}
	if ports1 != ports2 {
		c.Fatalf("Ports must be same: %s != %s", ports1, ports2)
	}
}

func (s *DockerSuite) TestBuildFails(c *check.C) {
	name := "testbuildfails"
	_, err := buildImage(name,
		`FROM busybox
		RUN sh -c "exit 23"`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "returned a non-zero code: 23") {
			c.Fatalf("Wrong error %v, must be about non-zero code 23", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

func (s *DockerSuite) TestBuildFailsDockerfileEmpty(c *check.C) {
	name := "testbuildfails"
	_, err := buildImage(name, ``, true)
	if err != nil {
		if !strings.Contains(err.Error(), "The Dockerfile (Dockerfile) cannot be empty") {
			c.Fatalf("Wrong error %v, must be about empty Dockerfile", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

func (s *DockerSuite) TestBuildOnBuild(c *check.C) {
	name := "testbuildonbuild"
	_, err := buildImage(name,
		`FROM busybox
		ONBUILD RUN touch foobar`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	_, err = buildImage(name,
		fmt.Sprintf(`FROM %s
		RUN [ -f foobar ]`, name),
		true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildOnBuildForbiddenChained(c *check.C) {
	name := "testbuildonbuildforbiddenchained"
	_, err := buildImage(name,
		`FROM busybox
		ONBUILD ONBUILD RUN touch foobar`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "Chaining ONBUILD via `ONBUILD ONBUILD` isn't allowed") {
			c.Fatalf("Wrong error %v, must be about chaining ONBUILD", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

func (s *DockerSuite) TestBuildOnBuildForbiddenFrom(c *check.C) {
	name := "testbuildonbuildforbiddenfrom"
	_, err := buildImage(name,
		`FROM busybox
		ONBUILD FROM scratch`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "FROM isn't allowed as an ONBUILD trigger") {
			c.Fatalf("Wrong error %v, must be about FROM forbidden", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

func (s *DockerSuite) TestBuildOnBuildForbiddenMaintainer(c *check.C) {
	name := "testbuildonbuildforbiddenmaintainer"
	_, err := buildImage(name,
		`FROM busybox
		ONBUILD MAINTAINER docker.io`,
		true)
	if err != nil {
		if !strings.Contains(err.Error(), "MAINTAINER isn't allowed as an ONBUILD trigger") {
			c.Fatalf("Wrong error %v, must be about MAINTAINER forbidden", err)
		}
	} else {
		c.Fatal("Error must not be nil")
	}
}

// gh #2446
func (s *DockerSuite) TestBuildAddToSymlinkDest(c *check.C) {
	name := "testbuildaddtosymlinkdest"
	ctx, err := fakeContext(`FROM busybox
        RUN mkdir /foo
        RUN ln -s /foo /bar
        ADD foo /bar/
        RUN [ -f /bar/foo ]
        RUN [ -f /foo/foo ]`,
		map[string]string{
			"foo": "hello",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildEscapeWhitespace(c *check.C) {
	name := "testbuildescaping"

	_, err := buildImage(name, `
  FROM busybox
  MAINTAINER "Docker \
IO <io@\
docker.com>"
  `, true)
	if err != nil {
		c.Fatal(err)
	}

	res, err := inspectField(name, "Author")

	if err != nil {
		c.Fatal(err)
	}

	if res != "\"Docker IO <io@docker.com>\"" {
		c.Fatalf("Parsed string did not match the escaped string. Got: %q", res)
	}

}

func (s *DockerSuite) TestBuildVerifyIntString(c *check.C) {
	// Verify that strings that look like ints are still passed as strings
	name := "testbuildstringing"

	_, err := buildImage(name, `
  FROM busybox
  MAINTAINER 123
  `, true)

	if err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "inspect", name)

	if !strings.Contains(out, "\"123\"") {
		c.Fatalf("Output does not contain the int as a string:\n%s", out)
	}

}

func (s *DockerSuite) TestBuildDockerignore(c *check.C) {
	name := "testbuilddockerignore"
	dockerfile := `
        FROM busybox
        ADD . /bla
		RUN [[ -f /bla/src/x.go ]]
		RUN [[ -f /bla/Makefile ]]
		RUN [[ ! -e /bla/src/_vendor ]]
		RUN [[ ! -e /bla/.gitignore ]]
		RUN [[ ! -e /bla/README.md ]]
		RUN [[ ! -e /bla/dir/foo ]]
		RUN [[ ! -e /bla/foo ]]
		RUN [[ ! -e /bla/.git ]]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Makefile":         "all:",
		".git/HEAD":        "ref: foo",
		"src/x.go":         "package main",
		"src/_vendor/v.go": "package main",
		"dir/foo":          "",
		".gitignore":       "",
		"README.md":        "readme",
		".dockerignore": `
.git
pkg
.gitignore
src/_vendor
*.md
dir`,
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildDockerignoreCleanPaths(c *check.C) {
	name := "testbuilddockerignorecleanpaths"
	dockerfile := `
        FROM busybox
        ADD . /tmp/
        RUN (! ls /tmp/foo) && (! ls /tmp/foo2) && (! ls /tmp/dir1/foo)`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"foo":           "foo",
		"foo2":          "foo2",
		"dir1/foo":      "foo in dir1",
		".dockerignore": "./foo\ndir1//foo\n./dir1/../foo2",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildDockerignoreExceptions(c *check.C) {
	name := "testbuilddockerignoreexceptions"
	dockerfile := `
        FROM busybox
        ADD . /bla
		RUN [[ -f /bla/src/x.go ]]
		RUN [[ -f /bla/Makefile ]]
		RUN [[ ! -e /bla/src/_vendor ]]
		RUN [[ ! -e /bla/.gitignore ]]
		RUN [[ ! -e /bla/README.md ]]
		RUN [[  -e /bla/dir/dir/foo ]]
		RUN [[ ! -e /bla/dir/foo1 ]]
		RUN [[ -f /bla/dir/e ]]
		RUN [[ -f /bla/dir/e-dir/foo ]]
		RUN [[ ! -e /bla/foo ]]
		RUN [[ ! -e /bla/.git ]]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Makefile":         "all:",
		".git/HEAD":        "ref: foo",
		"src/x.go":         "package main",
		"src/_vendor/v.go": "package main",
		"dir/foo":          "",
		"dir/foo1":         "",
		"dir/dir/f1":       "",
		"dir/dir/foo":      "",
		"dir/e":            "",
		"dir/e-dir/foo":    "",
		".gitignore":       "",
		"README.md":        "readme",
		".dockerignore": `
.git
pkg
.gitignore
src/_vendor
*.md
dir
!dir/e*
!dir/dir/foo`,
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()
	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildDockerignoringDockerfile(c *check.C) {
	name := "testbuilddockerignoredockerfile"
	dockerfile := `
        FROM busybox
		ADD . /tmp/
		RUN ! ls /tmp/Dockerfile
		RUN ls /tmp/.dockerignore`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    dockerfile,
		".dockerignore": "Dockerfile\n",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't ignore Dockerfile correctly:%s", err)
	}

	// now try it with ./Dockerfile
	ctx.Add(".dockerignore", "./Dockerfile\n")
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't ignore ./Dockerfile correctly:%s", err)
	}

}

func (s *DockerSuite) TestBuildDockerignoringRenamedDockerfile(c *check.C) {
	name := "testbuilddockerignoredockerfile"
	dockerfile := `
        FROM busybox
		ADD . /tmp/
		RUN ls /tmp/Dockerfile
		RUN ! ls /tmp/MyDockerfile
		RUN ls /tmp/.dockerignore`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    "Should not use me",
		"MyDockerfile":  dockerfile,
		".dockerignore": "MyDockerfile\n",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't ignore MyDockerfile correctly:%s", err)
	}

	// now try it with ./MyDockerfile
	ctx.Add(".dockerignore", "./MyDockerfile\n")
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't ignore ./MyDockerfile correctly:%s", err)
	}

}

func (s *DockerSuite) TestBuildDockerignoringDockerignore(c *check.C) {
	name := "testbuilddockerignoredockerignore"
	dockerfile := `
        FROM busybox
		ADD . /tmp/
		RUN ! ls /tmp/.dockerignore
		RUN ls /tmp/Dockerfile`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    dockerfile,
		".dockerignore": ".dockerignore\n",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't ignore .dockerignore correctly:%s", err)
	}
}

func (s *DockerSuite) TestBuildDockerignoreTouchDockerfile(c *check.C) {
	var id1 string
	var id2 string

	name := "testbuilddockerignoretouchdockerfile"
	dockerfile := `
        FROM busybox
		ADD . /tmp/`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    dockerfile,
		".dockerignore": "Dockerfile\n",
	})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	if id1, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't build it correctly:%s", err)
	}

	if id2, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't build it correctly:%s", err)
	}
	if id1 != id2 {
		c.Fatalf("Didn't use the cache - 1")
	}

	// Now make sure touching Dockerfile doesn't invalidate the cache
	if err = ctx.Add("Dockerfile", dockerfile+"\n# hi"); err != nil {
		c.Fatalf("Didn't add Dockerfile: %s", err)
	}
	if id2, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't build it correctly:%s", err)
	}
	if id1 != id2 {
		c.Fatalf("Didn't use the cache - 2")
	}

	// One more time but just 'touch' it instead of changing the content
	if err = ctx.Add("Dockerfile", dockerfile+"\n# hi"); err != nil {
		c.Fatalf("Didn't add Dockerfile: %s", err)
	}
	if id2, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("Didn't build it correctly:%s", err)
	}
	if id1 != id2 {
		c.Fatalf("Didn't use the cache - 3")
	}

}

func (s *DockerSuite) TestBuildDockerignoringWholeDir(c *check.C) {
	name := "testbuilddockerignorewholedir"
	dockerfile := `
        FROM busybox
		COPY . /
		RUN [[ ! -e /.gitignore ]]
		RUN [[ -f /Makefile ]]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    "FROM scratch",
		"Makefile":      "all:",
		".gitignore":    "",
		".dockerignore": ".*\n",
	})
	c.Assert(err, check.IsNil)
	defer ctx.Close()
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

	c.Assert(ctx.Add(".dockerfile", "*"), check.IsNil)
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

	c.Assert(ctx.Add(".dockerfile", "."), check.IsNil)
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

	c.Assert(ctx.Add(".dockerfile", "?"), check.IsNil)
	if _, err = buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildDockerignoringBadExclusion(c *check.C) {
	name := "testbuilddockerignorewholedir"
	dockerfile := `
        FROM busybox
		COPY . /
		RUN [[ ! -e /.gitignore ]]
		RUN [[ -f /Makefile ]]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"Dockerfile":    "FROM scratch",
		"Makefile":      "all:",
		".gitignore":    "",
		".dockerignore": "!\n",
	})
	c.Assert(err, check.IsNil)
	defer ctx.Close()
	if _, err = buildImageFromContext(name, ctx, true); err == nil {
		c.Fatalf("Build was supposed to fail but didn't")
	}

	if err.Error() != "failed to build the image: Error checking context: 'Illegal exclusion pattern: !'.\n" {
		c.Fatalf("Incorrect output, got:%q", err.Error())
	}
}

func (s *DockerSuite) TestBuildLineBreak(c *check.C) {
	name := "testbuildlinebreak"
	_, err := buildImage(name,
		`FROM  busybox
RUN    sh -c 'echo root:testpass \
	> /tmp/passwd'
RUN    mkdir -p /var/run/sshd
RUN    [ "$(cat /tmp/passwd)" = "root:testpass" ]
RUN    [ "$(ls -d /var/run/sshd)" = "/var/run/sshd" ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildEOLInLine(c *check.C) {
	name := "testbuildeolinline"
	_, err := buildImage(name,
		`FROM   busybox
RUN    sh -c 'echo root:testpass > /tmp/passwd'
RUN    echo "foo \n bar"; echo "baz"
RUN    mkdir -p /var/run/sshd
RUN    [ "$(cat /tmp/passwd)" = "root:testpass" ]
RUN    [ "$(ls -d /var/run/sshd)" = "/var/run/sshd" ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildCommentsShebangs(c *check.C) {
	name := "testbuildcomments"
	_, err := buildImage(name,
		`FROM busybox
# This is an ordinary comment.
RUN { echo '#!/bin/sh'; echo 'echo hello world'; } > /hello.sh
RUN [ ! -x /hello.sh ]
# comment with line break \
RUN chmod +x /hello.sh
RUN [ -x /hello.sh ]
RUN [ "$(cat /hello.sh)" = $'#!/bin/sh\necho hello world' ]
RUN [ "$(/hello.sh)" = "hello world" ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildUsersAndGroups(c *check.C) {
	name := "testbuildusers"
	_, err := buildImage(name,
		`FROM busybox

# Make sure our defaults work
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)" = '0:0/root:root' ]

# TODO decide if "args.user = strconv.Itoa(syscall.Getuid())" is acceptable behavior for changeUser in sysvinit instead of "return nil" when "USER" isn't specified (so that we get the proper group list even if that is the empty list, even in the default case of not supplying an explicit USER to run as, which implies USER 0)
USER root
RUN [ "$(id -G):$(id -Gn)" = '0 10:root wheel' ]

# Setup dockerio user and group
RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
RUN echo 'dockerio:x:1001:' >> /etc/group

# Make sure we can switch to our user and all the information is exactly as we expect it to be
USER dockerio
RUN id -G
RUN id -Gn
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001:dockerio' ]

# Switch back to root and double check that worked exactly as we might expect it to
USER root
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '0:0/root:root/0 10:root wheel' ]

# Add a "supplementary" group for our dockerio user
RUN echo 'supplementary:x:1002:dockerio' >> /etc/group

# ... and then go verify that we get it like we expect
USER dockerio
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001 1002:dockerio supplementary' ]
USER 1001
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001 1002:dockerio supplementary' ]

# super test the new "user:group" syntax
USER dockerio:dockerio
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001:dockerio' ]
USER 1001:dockerio
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001:dockerio' ]
USER dockerio:1001
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001:dockerio' ]
USER 1001:1001
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1001/dockerio:dockerio/1001:dockerio' ]
USER dockerio:supplementary
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1002/dockerio:supplementary/1002:supplementary' ]
USER dockerio:1002
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1002/dockerio:supplementary/1002:supplementary' ]
USER 1001:supplementary
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1002/dockerio:supplementary/1002:supplementary' ]
USER 1001:1002
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1001:1002/dockerio:supplementary/1002:supplementary' ]

# make sure unknown uid/gid still works properly
USER 1042:1043
RUN [ "$(id -u):$(id -g)/$(id -un):$(id -gn)/$(id -G):$(id -Gn)" = '1042:1043/1042:1043/1043:1043' ]`,
		true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildEnvUsage(c *check.C) {
	name := "testbuildenvusage"
	dockerfile := `FROM busybox
ENV    HOME /root
ENV    PATH $HOME/bin:$PATH
ENV    PATH /tmp:$PATH
RUN    [ "$PATH" = "/tmp:$HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" ]
ENV    FOO /foo/baz
ENV    BAR /bar
ENV    BAZ $BAR
ENV    FOOPATH $PATH:$FOO
RUN    [ "$BAR" = "$BAZ" ]
RUN    [ "$FOOPATH" = "$PATH:/foo/baz" ]
ENV	   FROM hello/docker/world
ENV    TO /docker/world/hello
ADD    $FROM $TO
RUN    [ "$(cat $TO)" = "hello" ]
ENV    abc=def
ENV    ghi=$abc
RUN    [ "$ghi" = "def" ]
`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"hello/docker/world": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	_, err = buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildEnvUsage2(c *check.C) {
	name := "testbuildenvusage2"
	dockerfile := `FROM busybox
ENV    abc=def
RUN    [ "$abc" = "def" ]
ENV    def="hello world"
RUN    [ "$def" = "hello world" ]
ENV    def=hello\ world
RUN    [ "$def" = "hello world" ]
ENV    v1=abc v2="hi there"
RUN    [ "$v1" = "abc" ]
RUN    [ "$v2" = "hi there" ]
ENV    v3='boogie nights' v4="with'quotes too"
RUN    [ "$v3" = "boogie nights" ]
RUN    [ "$v4" = "with'quotes too" ]
ENV    abc=zzz FROM=hello/docker/world
ENV    abc=zzz TO=/docker/world/hello
ADD    $FROM $TO
RUN    [ "$(cat $TO)" = "hello" ]
ENV    abc "zzz"
RUN    [ $abc = "zzz" ]
ENV    abc 'yyy'
RUN    [ $abc = 'yyy' ]
ENV    abc=
RUN    [ "$abc" = "" ]

# use grep to make sure if the builder substitutes \$foo by mistake
# we don't get a false positive
ENV    abc=\$foo
RUN    [ "$abc" = "\$foo" ] && (echo "$abc" | grep foo)
ENV    abc \$foo
RUN    [ "$abc" = "\$foo" ] && (echo "$abc" | grep foo)

ENV    abc=\'foo\'
RUN    [ "$abc" = "'foo'" ]
ENV    abc=\"foo\"
RUN    [ "$abc" = "\"foo\"" ]
ENV    abc "foo"
RUN    [ "$abc" = "foo" ]
ENV    abc 'foo'
RUN    [ "$abc" = 'foo' ]
ENV    abc \'foo\'
RUN    [ "$abc" = "'foo'" ]
ENV    abc \"foo\"
RUN    [ "$abc" = '"foo"' ]

ENV    abc=ABC
RUN    [ "$abc" = "ABC" ]
ENV    def=${abc:-DEF}
RUN    [ "$def" = "ABC" ]
ENV    def=${ccc:-DEF}
RUN    [ "$def" = "DEF" ]
ENV    def=${ccc:-${def}xx}
RUN    [ "$def" = "DEFxx" ]
ENV    def=${def:+ALT}
RUN    [ "$def" = "ALT" ]
ENV    def=${def:+${abc}:}
RUN    [ "$def" = "ABC:" ]
ENV    def=${ccc:-\$abc:}
RUN    [ "$def" = '$abc:' ]
ENV    def=${ccc:-\${abc}:}
RUN    [ "$def" = '${abc:}' ]
ENV    mypath=${mypath:+$mypath:}/home
RUN    [ "$mypath" = '/home' ]
ENV    mypath=${mypath:+$mypath:}/away
RUN    [ "$mypath" = '/home:/away' ]

ENV    e1=bar
ENV    e2=$e1
ENV    e3=$e11
ENV    e4=\$e1
ENV    e5=\$e11
RUN    [ "$e0,$e1,$e2,$e3,$e4,$e5" = ',bar,bar,,$e1,$e11' ]

ENV    ee1 bar
ENV    ee2 $ee1
ENV    ee3 $ee11
ENV    ee4 \$ee1
ENV    ee5 \$ee11
RUN    [ "$ee1,$ee2,$ee3,$ee4,$ee5" = 'bar,bar,,$ee1,$ee11' ]

ENV    eee1="foo"
ENV    eee2='foo'
ENV    eee3 "foo"
ENV    eee4 'foo'
RUN    [ "$eee1,$eee2,$eee3,$eee4" = 'foo,foo,foo,foo' ]

`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"hello/docker/world": "hello",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	_, err = buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddScript(c *check.C) {
	name := "testbuildaddscript"
	dockerfile := `
FROM busybox
ADD test /test
RUN ["chmod","+x","/test"]
RUN ["/test"]
RUN [ "$(cat /testfile)" = 'test!' ]`
	ctx, err := fakeContext(dockerfile, map[string]string{
		"test": "#!/bin/sh\necho 'test!' > /testfile",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	_, err = buildImageFromContext(name, ctx, true)
	if err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestBuildAddTar(c *check.C) {
	name := "testbuildaddtar"

	ctx := func() *FakeContext {
		dockerfile := `
FROM busybox
ADD test.tar /
RUN cat /test/foo | grep Hi
ADD test.tar /test.tar
RUN cat /test.tar/test/foo | grep Hi
ADD test.tar /unlikely-to-exist
RUN cat /unlikely-to-exist/test/foo | grep Hi
ADD test.tar /unlikely-to-exist-trailing-slash/
RUN cat /unlikely-to-exist-trailing-slash/test/foo | grep Hi
RUN mkdir /existing-directory
ADD test.tar /existing-directory
RUN cat /existing-directory/test/foo | grep Hi
ADD test.tar /existing-directory-trailing-slash/
RUN cat /existing-directory-trailing-slash/test/foo | grep Hi`
		tmpDir, err := ioutil.TempDir("", "fake-context")
		testTar, err := os.Create(filepath.Join(tmpDir, "test.tar"))
		if err != nil {
			c.Fatalf("failed to create test.tar archive: %v", err)
		}
		defer testTar.Close()

		tw := tar.NewWriter(testTar)

		if err := tw.WriteHeader(&tar.Header{
			Name: "test/foo",
			Size: 2,
		}); err != nil {
			c.Fatalf("failed to write tar file header: %v", err)
		}
		if _, err := tw.Write([]byte("Hi")); err != nil {
			c.Fatalf("failed to write tar file content: %v", err)
		}
		if err := tw.Close(); err != nil {
			c.Fatalf("failed to close tar archive: %v", err)
		}

		if err := ioutil.WriteFile(filepath.Join(tmpDir, "Dockerfile"), []byte(dockerfile), 0644); err != nil {
			c.Fatalf("failed to open destination dockerfile: %v", err)
		}
		return fakeContextFromDir(tmpDir)
	}()
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("build failed to complete for TestBuildAddTar: %v", err)
	}

}

func (s *DockerSuite) TestBuildAddTarXz(c *check.C) {
	name := "testbuildaddtarxz"

	ctx := func() *FakeContext {
		dockerfile := `
			FROM busybox
			ADD test.tar.xz /
			RUN cat /test/foo | grep Hi`
		tmpDir, err := ioutil.TempDir("", "fake-context")
		testTar, err := os.Create(filepath.Join(tmpDir, "test.tar"))
		if err != nil {
			c.Fatalf("failed to create test.tar archive: %v", err)
		}
		defer testTar.Close()

		tw := tar.NewWriter(testTar)

		if err := tw.WriteHeader(&tar.Header{
			Name: "test/foo",
			Size: 2,
		}); err != nil {
			c.Fatalf("failed to write tar file header: %v", err)
		}
		if _, err := tw.Write([]byte("Hi")); err != nil {
			c.Fatalf("failed to write tar file content: %v", err)
		}
		if err := tw.Close(); err != nil {
			c.Fatalf("failed to close tar archive: %v", err)
		}

		xzCompressCmd := exec.Command("xz", "-k", "test.tar")
		xzCompressCmd.Dir = tmpDir
		out, _, err := runCommandWithOutput(xzCompressCmd)
		if err != nil {
			c.Fatal(err, out)
		}

		if err := ioutil.WriteFile(filepath.Join(tmpDir, "Dockerfile"), []byte(dockerfile), 0644); err != nil {
			c.Fatalf("failed to open destination dockerfile: %v", err)
		}
		return fakeContextFromDir(tmpDir)
	}()

	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("build failed to complete for TestBuildAddTarXz: %v", err)
	}

}

func (s *DockerSuite) TestBuildAddTarXzGz(c *check.C) {
	name := "testbuildaddtarxzgz"

	ctx := func() *FakeContext {
		dockerfile := `
			FROM busybox
			ADD test.tar.xz.gz /
			RUN ls /test.tar.xz.gz`
		tmpDir, err := ioutil.TempDir("", "fake-context")
		testTar, err := os.Create(filepath.Join(tmpDir, "test.tar"))
		if err != nil {
			c.Fatalf("failed to create test.tar archive: %v", err)
		}
		defer testTar.Close()

		tw := tar.NewWriter(testTar)

		if err := tw.WriteHeader(&tar.Header{
			Name: "test/foo",
			Size: 2,
		}); err != nil {
			c.Fatalf("failed to write tar file header: %v", err)
		}
		if _, err := tw.Write([]byte("Hi")); err != nil {
			c.Fatalf("failed to write tar file content: %v", err)
		}
		if err := tw.Close(); err != nil {
			c.Fatalf("failed to close tar archive: %v", err)
		}

		xzCompressCmd := exec.Command("xz", "-k", "test.tar")
		xzCompressCmd.Dir = tmpDir
		out, _, err := runCommandWithOutput(xzCompressCmd)
		if err != nil {
			c.Fatal(err, out)
		}

		gzipCompressCmd := exec.Command("gzip", "test.tar.xz")
		gzipCompressCmd.Dir = tmpDir
		out, _, err = runCommandWithOutput(gzipCompressCmd)
		if err != nil {
			c.Fatal(err, out)
		}

		if err := ioutil.WriteFile(filepath.Join(tmpDir, "Dockerfile"), []byte(dockerfile), 0644); err != nil {
			c.Fatalf("failed to open destination dockerfile: %v", err)
		}
		return fakeContextFromDir(tmpDir)
	}()

	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatalf("build failed to complete for TestBuildAddTarXz: %v", err)
	}

}

func (s *DockerSuite) TestBuildFromGIT(c *check.C) {
	name := "testbuildfromgit"
	git, err := fakeGIT("repo", map[string]string{
		"Dockerfile": `FROM busybox
					ADD first /first
					RUN [ -f /first ]
					MAINTAINER docker`,
		"first": "test git data",
	}, true)
	if err != nil {
		c.Fatal(err)
	}
	defer git.Close()

	_, err = buildImageFromPath(name, git.RepoURL, true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Author")
	if err != nil {
		c.Fatal(err)
	}
	if res != "docker" {
		c.Fatalf("Maintainer should be docker, got %s", res)
	}
}

func (s *DockerSuite) TestBuildFromGITWithContext(c *check.C) {
	name := "testbuildfromgit"
	git, err := fakeGIT("repo", map[string]string{
		"docker/Dockerfile": `FROM busybox
					ADD first /first
					RUN [ -f /first ]
					MAINTAINER docker`,
		"docker/first": "test git data",
	}, true)
	if err != nil {
		c.Fatal(err)
	}
	defer git.Close()

	u := fmt.Sprintf("%s#master:docker", git.RepoURL)
	_, err = buildImageFromPath(name, u, true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Author")
	if err != nil {
		c.Fatal(err)
	}
	if res != "docker" {
		c.Fatalf("Maintainer should be docker, got %s", res)
	}
}

func (s *DockerSuite) TestBuildFromRemoteTarball(c *check.C) {
	name := "testbuildfromremotetarball"

	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte(`FROM busybox
					MAINTAINER docker`)
	if err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if _, err := tw.Write(dockerfile); err != nil {
		c.Fatalf("failed to write tar file content: %v", err)
	}
	if err := tw.Close(); err != nil {
		c.Fatalf("failed to close tar archive: %v", err)
	}

	server, err := fakeBinaryStorage(map[string]*bytes.Buffer{
		"testT.tar": buffer,
	})
	c.Assert(err, check.IsNil)

	defer server.Close()

	_, err = buildImageFromPath(name, server.URL()+"/testT.tar", true)
	c.Assert(err, check.IsNil)

	res, err := inspectField(name, "Author")
	c.Assert(err, check.IsNil)

	if res != "docker" {
		c.Fatalf("Maintainer should be docker, got %s", res)
	}
}

func (s *DockerSuite) TestBuildCleanupCmdOnEntrypoint(c *check.C) {
	name := "testbuildcmdcleanuponentrypoint"
	if _, err := buildImage(name,
		`FROM scratch
        CMD ["test"]
		ENTRYPOINT ["echo"]`,
		true); err != nil {
		c.Fatal(err)
	}
	if _, err := buildImage(name,
		fmt.Sprintf(`FROM %s
		ENTRYPOINT ["cat"]`, name),
		true); err != nil {
		c.Fatal(err)
	}
	res, err := inspectField(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if res != "<nil>" {
		c.Fatalf("Cmd %s, expected nil", res)
	}

	res, err = inspectField(name, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err)
	}
	if expected := "{[cat]}"; res != expected {
		c.Fatalf("Entrypoint %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildClearCmd(c *check.C) {
	name := "testbuildclearcmd"
	_, err := buildImage(name,
		`From scratch
   ENTRYPOINT ["/bin/bash"]
   CMD []`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectFieldJSON(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if res != "[]" {
		c.Fatalf("Cmd %s, expected %s", res, "[]")
	}
}

func (s *DockerSuite) TestBuildEmptyCmd(c *check.C) {
	name := "testbuildemptycmd"
	if _, err := buildImage(name, "FROM scratch\nMAINTAINER quux\n", true); err != nil {
		c.Fatal(err)
	}
	res, err := inspectFieldJSON(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if res != "null" {
		c.Fatalf("Cmd %s, expected %s", res, "null")
	}
}

func (s *DockerSuite) TestBuildOnBuildOutput(c *check.C) {
	name := "testbuildonbuildparent"
	if _, err := buildImage(name, "FROM busybox\nONBUILD RUN echo foo\n", true); err != nil {
		c.Fatal(err)
	}

	_, out, err := buildImageWithOut(name, "FROM "+name+"\nMAINTAINER quux\n", true)
	if err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(out, "Trigger 0, RUN echo foo") {
		c.Fatal("failed to find the ONBUILD output", out)
	}

}

func (s *DockerSuite) TestBuildInvalidTag(c *check.C) {
	name := "abcd:" + stringutils.GenerateRandomAlphaOnlyString(200)
	_, out, err := buildImageWithOut(name, "FROM scratch\nMAINTAINER quux\n", true)
	// if the error doesnt check for illegal tag name, or the image is built
	// then this should fail
	if !strings.Contains(out, "Illegal tag name") || strings.Contains(out, "Sending build context to Docker daemon") {
		c.Fatalf("failed to stop before building. Error: %s, Output: %s", err, out)
	}
}

func (s *DockerSuite) TestBuildCmdShDashC(c *check.C) {
	name := "testbuildcmdshc"
	if _, err := buildImage(name, "FROM busybox\nCMD echo cmd\n", true); err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err, res)
	}

	expected := `["/bin/sh","-c","echo cmd"]`

	if res != expected {
		c.Fatalf("Expected value %s not in Config.Cmd: %s", expected, res)
	}

}

func (s *DockerSuite) TestBuildCmdSpaces(c *check.C) {
	// Test to make sure that when we strcat arrays we take into account
	// the arg separator to make sure ["echo","hi"] and ["echo hi"] don't
	// look the same
	name := "testbuildcmdspaces"
	var id1 string
	var id2 string
	var err error

	if id1, err = buildImage(name, "FROM busybox\nCMD [\"echo hi\"]\n", true); err != nil {
		c.Fatal(err)
	}

	if id2, err = buildImage(name, "FROM busybox\nCMD [\"echo\", \"hi\"]\n", true); err != nil {
		c.Fatal(err)
	}

	if id1 == id2 {
		c.Fatal("Should not have resulted in the same CMD")
	}

	// Now do the same with ENTRYPOINT
	if id1, err = buildImage(name, "FROM busybox\nENTRYPOINT [\"echo hi\"]\n", true); err != nil {
		c.Fatal(err)
	}

	if id2, err = buildImage(name, "FROM busybox\nENTRYPOINT [\"echo\", \"hi\"]\n", true); err != nil {
		c.Fatal(err)
	}

	if id1 == id2 {
		c.Fatal("Should not have resulted in the same ENTRYPOINT")
	}

}

func (s *DockerSuite) TestBuildCmdJSONNoShDashC(c *check.C) {
	name := "testbuildcmdjson"
	if _, err := buildImage(name, "FROM busybox\nCMD [\"echo\", \"cmd\"]", true); err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name, "Config.Cmd")
	if err != nil {
		c.Fatal(err, res)
	}

	expected := `["echo","cmd"]`

	if res != expected {
		c.Fatalf("Expected value %s not in Config.Cmd: %s", expected, res)
	}

}

func (s *DockerSuite) TestBuildErrorInvalidInstruction(c *check.C) {
	name := "testbuildignoreinvalidinstruction"

	out, _, err := buildImageWithOut(name, "FROM busybox\nfoo bar", true)
	if err == nil {
		c.Fatalf("Should have failed: %s", out)
	}

}

func (s *DockerSuite) TestBuildEntrypointInheritance(c *check.C) {

	if _, err := buildImage("parent", `
    FROM busybox
    ENTRYPOINT exit 130
    `, true); err != nil {
		c.Fatal(err)
	}

	if _, status, _ := dockerCmdWithError(c, "run", "parent"); status != 130 {
		c.Fatalf("expected exit code 130 but received %d", status)
	}

	if _, err := buildImage("child", `
    FROM parent
    ENTRYPOINT exit 5
    `, true); err != nil {
		c.Fatal(err)
	}

	if _, status, _ := dockerCmdWithError(c, "run", "child"); status != 5 {
		c.Fatalf("expected exit code 5 but received %d", status)
	}

}

func (s *DockerSuite) TestBuildEntrypointInheritanceInspect(c *check.C) {
	var (
		name     = "testbuildepinherit"
		name2    = "testbuildepinherit2"
		expected = `["/bin/sh","-c","echo quux"]`
	)

	if _, err := buildImage(name, "FROM busybox\nENTRYPOINT /foo/bar", true); err != nil {
		c.Fatal(err)
	}

	if _, err := buildImage(name2, fmt.Sprintf("FROM %s\nENTRYPOINT echo quux", name), true); err != nil {
		c.Fatal(err)
	}

	res, err := inspectFieldJSON(name2, "Config.Entrypoint")
	if err != nil {
		c.Fatal(err, res)
	}

	if res != expected {
		c.Fatalf("Expected value %s not in Config.Entrypoint: %s", expected, res)
	}

	out, _ := dockerCmd(c, "run", "-t", name2)

	expected = "quux"

	if strings.TrimSpace(out) != expected {
		c.Fatalf("Expected output is %s, got %s", expected, out)
	}

}

func (s *DockerSuite) TestBuildRunShEntrypoint(c *check.C) {
	name := "testbuildentrypoint"
	_, err := buildImage(name,
		`FROM busybox
                                ENTRYPOINT /bin/echo`,
		true)
	if err != nil {
		c.Fatal(err)
	}

	dockerCmd(c, "run", "--rm", name)
}

func (s *DockerSuite) TestBuildExoticShellInterpolation(c *check.C) {
	name := "testbuildexoticshellinterpolation"

	_, err := buildImage(name, `
		FROM busybox

		ENV SOME_VAR a.b.c

		RUN [ "$SOME_VAR"       = 'a.b.c' ]
		RUN [ "${SOME_VAR}"     = 'a.b.c' ]
		RUN [ "${SOME_VAR%.*}"  = 'a.b'   ]
		RUN [ "${SOME_VAR%%.*}" = 'a'     ]
		RUN [ "${SOME_VAR#*.}"  = 'b.c'   ]
		RUN [ "${SOME_VAR##*.}" = 'c'     ]
		RUN [ "${SOME_VAR/c/d}" = 'a.b.d' ]
		RUN [ "${#SOME_VAR}"    = '5'     ]

		RUN [ "${SOME_UNSET_VAR:-$SOME_VAR}" = 'a.b.c' ]
		RUN [ "${SOME_VAR:+Version: ${SOME_VAR}}" = 'Version: a.b.c' ]
		RUN [ "${SOME_UNSET_VAR:+${SOME_VAR}}" = '' ]
		RUN [ "${SOME_UNSET_VAR:-${SOME_VAR:-d.e.f}}" = 'a.b.c' ]
	`, false)
	if err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildVerifySingleQuoteFails(c *check.C) {
	// This testcase is supposed to generate an error because the
	// JSON array we're passing in on the CMD uses single quotes instead
	// of double quotes (per the JSON spec). This means we interpret it
	// as a "string" insead of "JSON array" and pass it on to "sh -c" and
	// it should barf on it.
	name := "testbuildsinglequotefails"

	if _, err := buildImage(name,
		`FROM busybox
		CMD [ '/bin/sh', '-c', 'echo hi' ]`,
		true); err != nil {
		c.Fatal(err)
	}

	if _, _, err := dockerCmdWithError(c, "run", "--rm", name); err == nil {
		c.Fatal("The image was not supposed to be able to run")
	}

}

func (s *DockerSuite) TestBuildVerboseOut(c *check.C) {
	name := "testbuildverboseout"

	_, out, err := buildImageWithOut(name,
		`FROM busybox
RUN echo 123`,
		false)

	if err != nil {
		c.Fatal(err)
	}
	if !strings.Contains(out, "\n123\n") {
		c.Fatalf("Output should contain %q: %q", "123", out)
	}

}

func (s *DockerSuite) TestBuildWithTabs(c *check.C) {
	name := "testbuildwithtabs"
	_, err := buildImage(name,
		"FROM busybox\nRUN echo\tone\t\ttwo", true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectFieldJSON(name, "ContainerConfig.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	expected1 := `["/bin/sh","-c","echo\tone\t\ttwo"]`
	expected2 := `["/bin/sh","-c","echo\u0009one\u0009\u0009two"]` // syntactically equivalent, and what Go 1.3 generates
	if res != expected1 && res != expected2 {
		c.Fatalf("Missing tabs.\nGot: %s\nExp: %s or %s", res, expected1, expected2)
	}
}

func (s *DockerSuite) TestBuildLabels(c *check.C) {
	name := "testbuildlabel"
	expected := `{"License":"GPL","Vendor":"Acme"}`
	_, err := buildImage(name,
		`FROM busybox
		LABEL Vendor=Acme
                LABEL License GPL`,
		true)
	if err != nil {
		c.Fatal(err)
	}
	res, err := inspectFieldJSON(name, "Config.Labels")
	if err != nil {
		c.Fatal(err)
	}
	if res != expected {
		c.Fatalf("Labels %s, expected %s", res, expected)
	}
}

func (s *DockerSuite) TestBuildLabelsCache(c *check.C) {
	name := "testbuildlabelcache"

	id1, err := buildImage(name,
		`FROM busybox
		LABEL Vendor=Acme`, false)
	if err != nil {
		c.Fatalf("Build 1 should have worked: %v", err)
	}

	id2, err := buildImage(name,
		`FROM busybox
		LABEL Vendor=Acme`, true)
	if err != nil || id1 != id2 {
		c.Fatalf("Build 2 should have worked & used cache(%s,%s): %v", id1, id2, err)
	}

	id2, err = buildImage(name,
		`FROM busybox
		LABEL Vendor=Acme1`, true)
	if err != nil || id1 == id2 {
		c.Fatalf("Build 3 should have worked & NOT used cache(%s,%s): %v", id1, id2, err)
	}

	id2, err = buildImage(name,
		`FROM busybox
		LABEL Vendor Acme`, true) // Note: " " and "=" should be same
	if err != nil || id1 != id2 {
		c.Fatalf("Build 4 should have worked & used cache(%s,%s): %v", id1, id2, err)
	}

	// Now make sure the cache isn't used by mistake
	id1, err = buildImage(name,
		`FROM busybox
       LABEL f1=b1 f2=b2`, false)
	if err != nil {
		c.Fatalf("Build 5 should have worked: %q", err)
	}

	id2, err = buildImage(name,
		`FROM busybox
       LABEL f1="b1 f2=b2"`, true)
	if err != nil || id1 == id2 {
		c.Fatalf("Build 6 should have worked & NOT used the cache(%s,%s): %q", id1, id2, err)
	}

}

func (s *DockerSuite) TestBuildStderr(c *check.C) {
	// This test just makes sure that no non-error output goes
	// to stderr
	name := "testbuildstderr"
	_, _, stderr, err := buildImageWithStdoutStderr(name,
		"FROM busybox\nRUN echo one", true)
	if err != nil {
		c.Fatal(err)
	}

	if runtime.GOOS == "windows" {
		// stderr might contain a security warning on windows
		lines := strings.Split(stderr, "\n")
		for _, v := range lines {
			if v != "" && !strings.Contains(v, "SECURITY WARNING:") {
				c.Fatalf("Stderr contains unexpected output line: %q", v)
			}
		}
	} else {
		if stderr != "" {
			c.Fatalf("Stderr should have been empty, instead its: %q", stderr)
		}
	}
}

func (s *DockerSuite) TestBuildChownSingleFile(c *check.C) {
	testRequires(c, UnixCli) // test uses chown: not available on windows

	name := "testbuildchownsinglefile"

	ctx, err := fakeContext(`
FROM busybox
COPY test /
RUN ls -l /test
RUN [ $(ls -l /test | awk '{print $3":"$4}') = 'root:root' ]
`, map[string]string{
		"test": "test",
	})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if err := os.Chown(filepath.Join(ctx.Dir, "test"), 4242, 4242); err != nil {
		c.Fatal(err)
	}

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildSymlinkBreakout(c *check.C) {
	name := "testbuildsymlinkbreakout"
	tmpdir, err := ioutil.TempDir("", name)
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	ctx := filepath.Join(tmpdir, "context")
	if err := os.MkdirAll(ctx, 0755); err != nil {
		c.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(ctx, "Dockerfile"), []byte(`
	from busybox
	add symlink.tar /
	add inject /symlink/
	`), 0644); err != nil {
		c.Fatal(err)
	}
	inject := filepath.Join(ctx, "inject")
	if err := ioutil.WriteFile(inject, nil, 0644); err != nil {
		c.Fatal(err)
	}
	f, err := os.Create(filepath.Join(ctx, "symlink.tar"))
	if err != nil {
		c.Fatal(err)
	}
	w := tar.NewWriter(f)
	w.WriteHeader(&tar.Header{
		Name:     "symlink2",
		Typeflag: tar.TypeSymlink,
		Linkname: "/../../../../../../../../../../../../../../",
		Uid:      os.Getuid(),
		Gid:      os.Getgid(),
	})
	w.WriteHeader(&tar.Header{
		Name:     "symlink",
		Typeflag: tar.TypeSymlink,
		Linkname: filepath.Join("symlink2", tmpdir),
		Uid:      os.Getuid(),
		Gid:      os.Getgid(),
	})
	w.Close()
	f.Close()
	if _, err := buildImageFromContext(name, fakeContextFromDir(ctx), false); err != nil {
		c.Fatal(err)
	}
	if _, err := os.Lstat(filepath.Join(tmpdir, "inject")); err == nil {
		c.Fatal("symlink breakout - inject")
	} else if !os.IsNotExist(err) {
		c.Fatalf("unexpected error: %v", err)
	}
}

func (s *DockerSuite) TestBuildXZHost(c *check.C) {
	name := "testbuildxzhost"

	ctx, err := fakeContext(`
FROM busybox
ADD xz /usr/local/sbin/
RUN chmod 755 /usr/local/sbin/xz
ADD test.xz /
RUN [ ! -e /injected ]`,
		map[string]string{
			"test.xz": "\xfd\x37\x7a\x58\x5a\x00\x00\x04\xe6\xd6\xb4\x46\x02\x00" +
				"\x21\x01\x16\x00\x00\x00\x74\x2f\xe5\xa3\x01\x00\x3f\xfd" +
				"\x37\x7a\x58\x5a\x00\x00\x04\xe6\xd6\xb4\x46\x02\x00\x21",
			"xz": "#!/bin/sh\ntouch /injected",
		})

	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, true); err != nil {
		c.Fatal(err)
	}

}

func (s *DockerSuite) TestBuildVolumesRetainContents(c *check.C) {
	var (
		name     = "testbuildvolumescontent"
		expected = "some text"
	)
	ctx, err := fakeContext(`
FROM busybox
COPY content /foo/file
VOLUME /foo
CMD cat /foo/file`,
		map[string]string{
			"content": expected,
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err := buildImageFromContext(name, ctx, false); err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "run", "--rm", name)
	if out != expected {
		c.Fatalf("expected file contents for /foo/file to be %q but received %q", expected, out)
	}

}

func (s *DockerSuite) TestBuildRenamedDockerfile(c *check.C) {

	ctx, err := fakeContext(`FROM busybox
	RUN echo from Dockerfile`,
		map[string]string{
			"Dockerfile":       "FROM busybox\nRUN echo from Dockerfile",
			"files/Dockerfile": "FROM busybox\nRUN echo from files/Dockerfile",
			"files/dFile":      "FROM busybox\nRUN echo from files/dFile",
			"dFile":            "FROM busybox\nRUN echo from dFile",
			"files/dFile2":     "FROM busybox\nRUN echo from files/dFile2",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-t", "test1", ".")
	if err != nil {
		c.Fatalf("Failed to build: %s\n%s", out, err)
	}
	if !strings.Contains(out, "from Dockerfile") {
		c.Fatalf("test1 should have used Dockerfile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, ctx.Dir, "build", "-f", filepath.Join("files", "Dockerfile"), "-t", "test2", ".")
	if err != nil {
		c.Fatal(err)
	}
	if !strings.Contains(out, "from files/Dockerfile") {
		c.Fatalf("test2 should have used files/Dockerfile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, ctx.Dir, "build", fmt.Sprintf("--file=%s", filepath.Join("files", "dFile")), "-t", "test3", ".")
	if err != nil {
		c.Fatal(err)
	}
	if !strings.Contains(out, "from files/dFile") {
		c.Fatalf("test3 should have used files/dFile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, ctx.Dir, "build", "--file=dFile", "-t", "test4", ".")
	if err != nil {
		c.Fatal(err)
	}
	if !strings.Contains(out, "from dFile") {
		c.Fatalf("test4 should have used dFile, output:%s", out)
	}

	dirWithNoDockerfile, _ := ioutil.TempDir(os.TempDir(), "test5")
	nonDockerfileFile := filepath.Join(dirWithNoDockerfile, "notDockerfile")
	if _, err = os.Create(nonDockerfileFile); err != nil {
		c.Fatal(err)
	}
	out, _, err = dockerCmdInDir(c, ctx.Dir, "build", fmt.Sprintf("--file=%s", nonDockerfileFile), "-t", "test5", ".")

	if err == nil {
		c.Fatalf("test5 was supposed to fail to find passwd")
	}

	if expected := fmt.Sprintf("The Dockerfile (%s) must be within the build context (.)", nonDockerfileFile); !strings.Contains(out, expected) {
		c.Fatalf("wrong error messsage:%v\nexpected to contain=%v", out, expected)
	}

	out, _, err = dockerCmdInDir(c, filepath.Join(ctx.Dir, "files"), "build", "-f", filepath.Join("..", "Dockerfile"), "-t", "test6", "..")
	if err != nil {
		c.Fatalf("test6 failed: %s", err)
	}
	if !strings.Contains(out, "from Dockerfile") {
		c.Fatalf("test6 should have used root Dockerfile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, filepath.Join(ctx.Dir, "files"), "build", "-f", filepath.Join(ctx.Dir, "files", "Dockerfile"), "-t", "test7", "..")
	if err != nil {
		c.Fatalf("test7 failed: %s", err)
	}
	if !strings.Contains(out, "from files/Dockerfile") {
		c.Fatalf("test7 should have used files Dockerfile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, filepath.Join(ctx.Dir, "files"), "build", "-f", filepath.Join("..", "Dockerfile"), "-t", "test8", ".")
	if err == nil || !strings.Contains(out, "must be within the build context") {
		c.Fatalf("test8 should have failed with Dockerfile out of context: %s", err)
	}

	tmpDir := os.TempDir()
	out, _, err = dockerCmdInDir(c, tmpDir, "build", "-t", "test9", ctx.Dir)
	if err != nil {
		c.Fatalf("test9 - failed: %s", err)
	}
	if !strings.Contains(out, "from Dockerfile") {
		c.Fatalf("test9 should have used root Dockerfile, output:%s", out)
	}

	out, _, err = dockerCmdInDir(c, filepath.Join(ctx.Dir, "files"), "build", "-f", "dFile2", "-t", "test10", ".")
	if err != nil {
		c.Fatalf("test10 should have worked: %s", err)
	}
	if !strings.Contains(out, "from files/dFile2") {
		c.Fatalf("test10 should have used files/dFile2, output:%s", out)
	}

}

func (s *DockerSuite) TestBuildFromMixedcaseDockerfile(c *check.C) {
	testRequires(c, UnixCli) // Dockerfile overwrites dockerfile on windows

	ctx, err := fakeContext(`FROM busybox
	RUN echo from dockerfile`,
		map[string]string{
			"dockerfile": "FROM busybox\nRUN echo from dockerfile",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-t", "test1", ".")
	if err != nil {
		c.Fatalf("Failed to build: %s\n%s", out, err)
	}

	if !strings.Contains(out, "from dockerfile") {
		c.Fatalf("Missing proper output: %s", out)
	}

}

func (s *DockerSuite) TestBuildWithTwoDockerfiles(c *check.C) {
	testRequires(c, UnixCli) // Dockerfile overwrites dockerfile on windows

	ctx, err := fakeContext(`FROM busybox
RUN echo from Dockerfile`,
		map[string]string{
			"dockerfile": "FROM busybox\nRUN echo from dockerfile",
		})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-t", "test1", ".")
	if err != nil {
		c.Fatalf("Failed to build: %s\n%s", out, err)
	}

	if !strings.Contains(out, "from Dockerfile") {
		c.Fatalf("Missing proper output: %s", out)
	}

}

func (s *DockerSuite) TestBuildFromURLWithF(c *check.C) {

	server, err := fakeStorage(map[string]string{"baz": `FROM busybox
RUN echo from baz
COPY * /tmp/
RUN find /tmp/`})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	ctx, err := fakeContext(`FROM busybox
RUN echo from Dockerfile`,
		map[string]string{})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	// Make sure that -f is ignored and that we don't use the Dockerfile
	// that's in the current dir
	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "-f", "baz", "-t", "test1", server.URL()+"/baz")
	if err != nil {
		c.Fatalf("Failed to build: %s\n%s", out, err)
	}

	if !strings.Contains(out, "from baz") ||
		strings.Contains(out, "/tmp/baz") ||
		!strings.Contains(out, "/tmp/Dockerfile") {
		c.Fatalf("Missing proper output: %s", out)
	}

}

func (s *DockerSuite) TestBuildFromStdinWithF(c *check.C) {

	ctx, err := fakeContext(`FROM busybox
RUN echo from Dockerfile`,
		map[string]string{})
	defer ctx.Close()
	if err != nil {
		c.Fatal(err)
	}

	// Make sure that -f is ignored and that we don't use the Dockerfile
	// that's in the current dir
	dockerCommand := exec.Command(dockerBinary, "build", "-f", "baz", "-t", "test1", "-")
	dockerCommand.Dir = ctx.Dir
	dockerCommand.Stdin = strings.NewReader(`FROM busybox
RUN echo from baz
COPY * /tmp/
RUN find /tmp/`)
	out, status, err := runCommandWithOutput(dockerCommand)
	if err != nil || status != 0 {
		c.Fatalf("Error building: %s", err)
	}

	if !strings.Contains(out, "from baz") ||
		strings.Contains(out, "/tmp/baz") ||
		!strings.Contains(out, "/tmp/Dockerfile") {
		c.Fatalf("Missing proper output: %s", out)
	}

}

func (s *DockerSuite) TestBuildFromOfficialNames(c *check.C) {
	name := "testbuildfromofficial"
	fromNames := []string{
		"busybox",
		"docker.io/busybox",
		"index.docker.io/busybox",
		"library/busybox",
		"docker.io/library/busybox",
		"index.docker.io/library/busybox",
	}
	for idx, fromName := range fromNames {
		imgName := fmt.Sprintf("%s%d", name, idx)
		_, err := buildImage(imgName, "FROM "+fromName, true)
		if err != nil {
			c.Errorf("Build failed using FROM %s: %s", fromName, err)
		}
		deleteImages(imgName)
	}
}

func (s *DockerSuite) TestBuildDockerfileOutsideContext(c *check.C) {
	testRequires(c, UnixCli) // uses os.Symlink: not implemented in windows at the time of writing (go-1.4.2)

	name := "testbuilddockerfileoutsidecontext"
	tmpdir, err := ioutil.TempDir("", name)
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	ctx := filepath.Join(tmpdir, "context")
	if err := os.MkdirAll(ctx, 0755); err != nil {
		c.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(ctx, "Dockerfile"), []byte("FROM scratch\nENV X Y"), 0644); err != nil {
		c.Fatal(err)
	}
	wd, err := os.Getwd()
	if err != nil {
		c.Fatal(err)
	}
	defer os.Chdir(wd)
	if err := os.Chdir(ctx); err != nil {
		c.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(tmpdir, "outsideDockerfile"), []byte("FROM scratch\nENV x y"), 0644); err != nil {
		c.Fatal(err)
	}
	if err := os.Symlink(filepath.Join("..", "outsideDockerfile"), filepath.Join(ctx, "dockerfile1")); err != nil {
		c.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(tmpdir, "outsideDockerfile"), filepath.Join(ctx, "dockerfile2")); err != nil {
		c.Fatal(err)
	}

	for _, dockerfilePath := range []string{
		filepath.Join("..", "outsideDockerfile"),
		filepath.Join(ctx, "dockerfile1"),
		filepath.Join(ctx, "dockerfile2"),
	} {
		out, _, err := dockerCmdWithError(c, "build", "-t", name, "--no-cache", "-f", dockerfilePath, ".")
		if err == nil {
			c.Fatalf("Expected error with %s. Out: %s", dockerfilePath, out)
		}
		if !strings.Contains(out, "must be within the build context") && !strings.Contains(out, "Cannot locate Dockerfile") {
			c.Fatalf("Unexpected error with %s. Out: %s", dockerfilePath, out)
		}
		deleteImages(name)
	}

	os.Chdir(tmpdir)

	// Path to Dockerfile should be resolved relative to working directory, not relative to context.
	// There is a Dockerfile in the context, but since there is no Dockerfile in the current directory, the following should fail
	out, _, err := dockerCmdWithError(c, "build", "-t", name, "--no-cache", "-f", "Dockerfile", ctx)
	if err == nil {
		c.Fatalf("Expected error. Out: %s", out)
	}
}

func (s *DockerSuite) TestBuildSpaces(c *check.C) {
	// Test to make sure that leading/trailing spaces on a command
	// doesn't change the error msg we get
	var (
		err1 error
		err2 error
	)

	name := "testspaces"
	ctx, err := fakeContext("FROM busybox\nCOPY\n",
		map[string]string{
			"Dockerfile": "FROM busybox\nCOPY\n",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err1 = buildImageFromContext(name, ctx, false); err1 == nil {
		c.Fatal("Build 1 was supposed to fail, but didn't")
	}

	ctx.Add("Dockerfile", "FROM busybox\nCOPY    ")
	if _, err2 = buildImageFromContext(name, ctx, false); err2 == nil {
		c.Fatal("Build 2 was supposed to fail, but didn't")
	}

	removeLogTimestamps := func(s string) string {
		return regexp.MustCompile(`time="(.*?)"`).ReplaceAllString(s, `time=[TIMESTAMP]`)
	}

	// Skip over the times
	e1 := removeLogTimestamps(err1.Error())
	e2 := removeLogTimestamps(err2.Error())

	// Ignore whitespace since that's what were verifying doesn't change stuff
	if strings.Replace(e1, " ", "", -1) != strings.Replace(e2, " ", "", -1) {
		c.Fatalf("Build 2's error wasn't the same as build 1's\n1:%s\n2:%s", err1, err2)
	}

	ctx.Add("Dockerfile", "FROM busybox\n   COPY")
	if _, err2 = buildImageFromContext(name, ctx, false); err2 == nil {
		c.Fatal("Build 3 was supposed to fail, but didn't")
	}

	// Skip over the times
	e1 = removeLogTimestamps(err1.Error())
	e2 = removeLogTimestamps(err2.Error())

	// Ignore whitespace since that's what were verifying doesn't change stuff
	if strings.Replace(e1, " ", "", -1) != strings.Replace(e2, " ", "", -1) {
		c.Fatalf("Build 3's error wasn't the same as build 1's\n1:%s\n3:%s", err1, err2)
	}

	ctx.Add("Dockerfile", "FROM busybox\n   COPY    ")
	if _, err2 = buildImageFromContext(name, ctx, false); err2 == nil {
		c.Fatal("Build 4 was supposed to fail, but didn't")
	}

	// Skip over the times
	e1 = removeLogTimestamps(err1.Error())
	e2 = removeLogTimestamps(err2.Error())

	// Ignore whitespace since that's what were verifying doesn't change stuff
	if strings.Replace(e1, " ", "", -1) != strings.Replace(e2, " ", "", -1) {
		c.Fatalf("Build 4's error wasn't the same as build 1's\n1:%s\n4:%s", err1, err2)
	}

}

func (s *DockerSuite) TestBuildSpacesWithQuotes(c *check.C) {
	// Test to make sure that spaces in quotes aren't lost
	name := "testspacesquotes"

	dockerfile := `FROM busybox
RUN echo "  \
  foo  "`

	_, out, err := buildImageWithOut(name, dockerfile, false)
	if err != nil {
		c.Fatal("Build failed:", err)
	}

	expecting := "\n    foo  \n"
	if !strings.Contains(out, expecting) {
		c.Fatalf("Bad output: %q expecting to contain %q", out, expecting)
	}

}

// #4393
func (s *DockerSuite) TestBuildVolumeFileExistsinContainer(c *check.C) {
	buildCmd := exec.Command(dockerBinary, "build", "-t", "docker-test-errcreatevolumewithfile", "-")
	buildCmd.Stdin = strings.NewReader(`
	FROM busybox
	RUN touch /foo
	VOLUME /foo
	`)

	out, _, err := runCommandWithOutput(buildCmd)
	if err == nil || !strings.Contains(out, "file exists") {
		c.Fatalf("expected build to fail when file exists in container at requested volume path")
	}

}

func (s *DockerSuite) TestBuildMissingArgs(c *check.C) {
	// Test to make sure that all Dockerfile commands (except the ones listed
	// in skipCmds) will generate an error if no args are provided.
	// Note: INSERT is deprecated so we exclude it because of that.
	skipCmds := map[string]struct{}{
		"CMD":        {},
		"RUN":        {},
		"ENTRYPOINT": {},
		"INSERT":     {},
	}

	for cmd := range command.Commands {
		cmd = strings.ToUpper(cmd)
		if _, ok := skipCmds[cmd]; ok {
			continue
		}

		var dockerfile string
		if cmd == "FROM" {
			dockerfile = cmd
		} else {
			// Add FROM to make sure we don't complain about it missing
			dockerfile = "FROM busybox\n" + cmd
		}

		ctx, err := fakeContext(dockerfile, map[string]string{})
		if err != nil {
			c.Fatal(err)
		}
		defer ctx.Close()
		var out string
		if out, err = buildImageFromContext("args", ctx, true); err == nil {
			c.Fatalf("%s was supposed to fail. Out:%s", cmd, out)
		}
		if !strings.Contains(err.Error(), cmd+" requires") {
			c.Fatalf("%s returned the wrong type of error:%s", cmd, err)
		}
	}

}

func (s *DockerSuite) TestBuildEmptyScratch(c *check.C) {
	_, out, err := buildImageWithOut("sc", "FROM scratch", true)
	if err == nil {
		c.Fatalf("Build was supposed to fail")
	}
	if !strings.Contains(out, "No image was generated") {
		c.Fatalf("Wrong error message: %v", out)
	}
}

func (s *DockerSuite) TestBuildDotDotFile(c *check.C) {
	ctx, err := fakeContext("FROM busybox\n",
		map[string]string{
			"..gitme": "",
		})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	if _, err = buildImageFromContext("sc", ctx, false); err != nil {
		c.Fatalf("Build was supposed to work: %s", err)
	}
}

func (s *DockerSuite) TestBuildNotVerbose(c *check.C) {

	ctx, err := fakeContext("FROM busybox\nENV abc=hi\nRUN echo $abc there", map[string]string{})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	// First do it w/verbose - baseline
	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "--no-cache", "-t", "verbose", ".")
	if err != nil {
		c.Fatalf("failed to build the image w/o -q: %s, %v", out, err)
	}
	if !strings.Contains(out, "hi there") {
		c.Fatalf("missing output:%s\n", out)
	}

	// Now do it w/o verbose
	out, _, err = dockerCmdInDir(c, ctx.Dir, "build", "--no-cache", "-q", "-t", "verbose", ".")
	if err != nil {
		c.Fatalf("failed to build the image w/ -q: %s, %v", out, err)
	}
	if strings.Contains(out, "hi there") {
		c.Fatalf("Bad output, should not contain 'hi there':%s", out)
	}

}

func (s *DockerSuite) TestBuildRUNoneJSON(c *check.C) {
	name := "testbuildrunonejson"

	ctx, err := fakeContext(`FROM hello-world:frozen
RUN [ "/hello" ]`, map[string]string{})
	if err != nil {
		c.Fatal(err)
	}
	defer ctx.Close()

	out, _, err := dockerCmdInDir(c, ctx.Dir, "build", "--no-cache", "-t", name, ".")
	if err != nil {
		c.Fatalf("failed to build the image: %s, %v", out, err)
	}

	if !strings.Contains(out, "Hello from Docker") {
		c.Fatalf("bad output: %s", out)
	}

}

func (s *DockerSuite) TestBuildEmptyStringVolume(c *check.C) {
	name := "testbuildemptystringvolume"

	_, err := buildImage(name, `
  FROM busybox
  ENV foo=""
  VOLUME $foo
  `, false)
	if err == nil {
		c.Fatal("Should have failed to build")
	}

}

func (s *DockerSuite) TestBuildContainerWithCgroupParent(c *check.C) {
	testRequires(c, NativeExecDriver)
	testRequires(c, SameHostDaemon)

	cgroupParent := "test"
	data, err := ioutil.ReadFile("/proc/self/cgroup")
	if err != nil {
		c.Fatalf("failed to read '/proc/self/cgroup - %v", err)
	}
	selfCgroupPaths := parseCgroupPaths(string(data))
	_, found := selfCgroupPaths["memory"]
	if !found {
		c.Fatalf("unable to find self cpu cgroup path. CgroupsPath: %v", selfCgroupPaths)
	}
	cmd := exec.Command(dockerBinary, "build", "--cgroup-parent", cgroupParent, "-")
	cmd.Stdin = strings.NewReader(`
FROM busybox
RUN cat /proc/self/cgroup
`)

	out, _, err := runCommandWithOutput(cmd)
	if err != nil {
		c.Fatalf("unexpected failure when running container with --cgroup-parent option - %s\n%v", string(out), err)
	}
}

func (s *DockerSuite) TestBuildNoDupOutput(c *check.C) {
	// Check to make sure our build output prints the Dockerfile cmd
	// property - there was a bug that caused it to be duplicated on the
	// Step X  line
	name := "testbuildnodupoutput"

	_, out, err := buildImageWithOut(name, `
  FROM busybox
  RUN env`, false)
	if err != nil {
		c.Fatalf("Build should have worked: %q", err)
	}

	exp := "\nStep 1 : RUN env\n"
	if !strings.Contains(out, exp) {
		c.Fatalf("Bad output\nGot:%s\n\nExpected to contain:%s\n", out, exp)
	}
}

func (s *DockerSuite) TestBuildBadCmdFlag(c *check.C) {
	name := "testbuildbadcmdflag"

	_, out, err := buildImageWithOut(name, `
  FROM busybox
  MAINTAINER --boo joe@example.com`, false)
	if err == nil {
		c.Fatal("Build should have failed")
	}

	exp := "\nUnknown flag: boo\n"
	if !strings.Contains(out, exp) {
		c.Fatalf("Bad output\nGot:%s\n\nExpected to contain:%s\n", out, exp)
	}
}

func (s *DockerSuite) TestBuildRUNErrMsg(c *check.C) {
	// Test to make sure the bad command is quoted with just "s and
	// not as a Go []string
	name := "testbuildbadrunerrmsg"
	_, out, err := buildImageWithOut(name, `
  FROM busybox
  RUN badEXE a1 \& a2	a3`, false) // tab between a2 and a3
	if err == nil {
		c.Fatal("Should have failed to build")
	}

	exp := `The command '/bin/sh -c badEXE a1 \& a2	a3' returned a non-zero code: 127`
	if !strings.Contains(out, exp) {
		c.Fatalf("RUN doesn't have the correct output:\nGot:%s\nExpected:%s", out, exp)
	}
}
