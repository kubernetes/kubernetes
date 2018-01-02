package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/configuration"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/manifest"
	"github.com/docker/distribution/manifest/manifestlist"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/factory"
	_ "github.com/docker/distribution/registry/storage/driver/testdriver"
	"github.com/docker/distribution/testutil"
	"github.com/docker/libtrust"
	"github.com/gorilla/handlers"
	"github.com/opencontainers/go-digest"
)

var headerConfig = http.Header{
	"X-Content-Type-Options": []string{"nosniff"},
}

const (
	// digestSha256EmptyTar is the canonical sha256 digest of empty data
	digestSha256EmptyTar = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// TestCheckAPI hits the base endpoint (/v2/) ensures we return the specified
// 200 OK response.
func TestCheckAPI(t *testing.T) {
	env := newTestEnv(t, false)
	defer env.Shutdown()
	baseURL, err := env.builder.BuildBaseURL()
	if err != nil {
		t.Fatalf("unexpected error building base url: %v", err)
	}

	resp, err := http.Get(baseURL)
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "issuing api base check", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Type":   []string{"application/json; charset=utf-8"},
		"Content-Length": []string{"2"},
	})

	p, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("unexpected error reading response body: %v", err)
	}

	if string(p) != "{}" {
		t.Fatalf("unexpected response body: %v", string(p))
	}
}

// TestCatalogAPI tests the /v2/_catalog endpoint
func TestCatalogAPI(t *testing.T) {
	chunkLen := 2
	env := newTestEnv(t, false)
	defer env.Shutdown()

	values := url.Values{
		"last": []string{""},
		"n":    []string{strconv.Itoa(chunkLen)}}

	catalogURL, err := env.builder.BuildCatalogURL(values)
	if err != nil {
		t.Fatalf("unexpected error building catalog url: %v", err)
	}

	// -----------------------------------
	// try to get an empty catalog
	resp, err := http.Get(catalogURL)
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "issuing catalog api check", resp, http.StatusOK)

	var ctlg struct {
		Repositories []string `json:"repositories"`
	}

	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(&ctlg); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	// we haven't pushed anything to the registry yet
	if len(ctlg.Repositories) != 0 {
		t.Fatalf("repositories has unexpected values")
	}

	if resp.Header.Get("Link") != "" {
		t.Fatalf("repositories has more data when none expected")
	}

	// -----------------------------------
	// push something to the registry and try again
	images := []string{"foo/aaaa", "foo/bbbb", "foo/cccc"}

	for _, image := range images {
		createRepository(env, t, image, "sometag")
	}

	resp, err = http.Get(catalogURL)
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "issuing catalog api check", resp, http.StatusOK)

	dec = json.NewDecoder(resp.Body)
	if err = dec.Decode(&ctlg); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	if len(ctlg.Repositories) != chunkLen {
		t.Fatalf("repositories has unexpected values")
	}

	for _, image := range images[:chunkLen] {
		if !contains(ctlg.Repositories, image) {
			t.Fatalf("didn't find our repository '%s' in the catalog", image)
		}
	}

	link := resp.Header.Get("Link")
	if link == "" {
		t.Fatalf("repositories has less data than expected")
	}

	newValues := checkLink(t, link, chunkLen, ctlg.Repositories[len(ctlg.Repositories)-1])

	// -----------------------------------
	// get the last chunk of data

	catalogURL, err = env.builder.BuildCatalogURL(newValues)
	if err != nil {
		t.Fatalf("unexpected error building catalog url: %v", err)
	}

	resp, err = http.Get(catalogURL)
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "issuing catalog api check", resp, http.StatusOK)

	dec = json.NewDecoder(resp.Body)
	if err = dec.Decode(&ctlg); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	if len(ctlg.Repositories) != 1 {
		t.Fatalf("repositories has unexpected values")
	}

	lastImage := images[len(images)-1]
	if !contains(ctlg.Repositories, lastImage) {
		t.Fatalf("didn't find our repository '%s' in the catalog", lastImage)
	}

	link = resp.Header.Get("Link")
	if link != "" {
		t.Fatalf("catalog has unexpected data")
	}
}

func checkLink(t *testing.T, urlStr string, numEntries int, last string) url.Values {
	re := regexp.MustCompile("<(/v2/_catalog.*)>; rel=\"next\"")
	matches := re.FindStringSubmatch(urlStr)

	if len(matches) != 2 {
		t.Fatalf("Catalog link address response was incorrect")
	}
	linkURL, _ := url.Parse(matches[1])
	urlValues := linkURL.Query()

	if urlValues.Get("n") != strconv.Itoa(numEntries) {
		t.Fatalf("Catalog link entry size is incorrect")
	}

	if urlValues.Get("last") != last {
		t.Fatal("Catalog link last entry is incorrect")
	}

	return urlValues
}

func contains(elems []string, e string) bool {
	for _, elem := range elems {
		if elem == e {
			return true
		}
	}
	return false
}

func TestURLPrefix(t *testing.T) {
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
	}
	config.HTTP.Prefix = "/test/"
	config.HTTP.Headers = headerConfig

	env := newTestEnvWithConfig(t, &config)
	defer env.Shutdown()

	baseURL, err := env.builder.BuildBaseURL()
	if err != nil {
		t.Fatalf("unexpected error building base url: %v", err)
	}

	parsed, _ := url.Parse(baseURL)
	if !strings.HasPrefix(parsed.Path, config.HTTP.Prefix) {
		t.Fatalf("Prefix %v not included in test url %v", config.HTTP.Prefix, baseURL)
	}

	resp, err := http.Get(baseURL)
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "issuing api base check", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Type":   []string{"application/json; charset=utf-8"},
		"Content-Length": []string{"2"},
	})
}

type blobArgs struct {
	imageName   reference.Named
	layerFile   io.ReadSeeker
	layerDigest digest.Digest
}

func makeBlobArgs(t *testing.T) blobArgs {
	layerFile, layerDigest, err := testutil.CreateRandomTarFile()
	if err != nil {
		t.Fatalf("error creating random layer file: %v", err)
	}

	args := blobArgs{
		layerFile:   layerFile,
		layerDigest: layerDigest,
	}
	args.imageName, _ = reference.WithName("foo/bar")
	return args
}

// TestBlobAPI conducts a full test of the of the blob api.
func TestBlobAPI(t *testing.T) {
	deleteEnabled := false
	env1 := newTestEnv(t, deleteEnabled)
	defer env1.Shutdown()
	args := makeBlobArgs(t)
	testBlobAPI(t, env1, args)

	deleteEnabled = true
	env2 := newTestEnv(t, deleteEnabled)
	defer env2.Shutdown()
	args = makeBlobArgs(t)
	testBlobAPI(t, env2, args)

}

func TestBlobDelete(t *testing.T) {
	deleteEnabled := true
	env := newTestEnv(t, deleteEnabled)
	defer env.Shutdown()

	args := makeBlobArgs(t)
	env = testBlobAPI(t, env, args)
	testBlobDelete(t, env, args)
}

func TestRelativeURL(t *testing.T) {
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
	}
	config.HTTP.Headers = headerConfig
	config.HTTP.RelativeURLs = false
	env := newTestEnvWithConfig(t, &config)
	defer env.Shutdown()
	ref, _ := reference.WithName("foo/bar")
	uploadURLBaseAbs, _ := startPushLayer(t, env, ref)

	u, err := url.Parse(uploadURLBaseAbs)
	if err != nil {
		t.Fatal(err)
	}
	if !u.IsAbs() {
		t.Fatal("Relative URL returned from blob upload chunk with non-relative configuration")
	}

	args := makeBlobArgs(t)
	resp, err := doPushLayer(t, env.builder, ref, args.layerDigest, uploadURLBaseAbs, args.layerFile)
	if err != nil {
		t.Fatalf("unexpected error doing layer push relative url: %v", err)
	}
	checkResponse(t, "relativeurl blob upload", resp, http.StatusCreated)
	u, err = url.Parse(resp.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	if !u.IsAbs() {
		t.Fatal("Relative URL returned from blob upload with non-relative configuration")
	}

	config.HTTP.RelativeURLs = true
	args = makeBlobArgs(t)
	uploadURLBaseRelative, _ := startPushLayer(t, env, ref)
	u, err = url.Parse(uploadURLBaseRelative)
	if err != nil {
		t.Fatal(err)
	}
	if u.IsAbs() {
		t.Fatal("Absolute URL returned from blob upload chunk with relative configuration")
	}

	// Start a new upload in absolute mode to get a valid base URL
	config.HTTP.RelativeURLs = false
	uploadURLBaseAbs, _ = startPushLayer(t, env, ref)
	u, err = url.Parse(uploadURLBaseAbs)
	if err != nil {
		t.Fatal(err)
	}
	if !u.IsAbs() {
		t.Fatal("Relative URL returned from blob upload chunk with non-relative configuration")
	}

	// Complete upload with relative URLs enabled to ensure the final location is relative
	config.HTTP.RelativeURLs = true
	resp, err = doPushLayer(t, env.builder, ref, args.layerDigest, uploadURLBaseAbs, args.layerFile)
	if err != nil {
		t.Fatalf("unexpected error doing layer push relative url: %v", err)
	}

	checkResponse(t, "relativeurl blob upload", resp, http.StatusCreated)
	u, err = url.Parse(resp.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	if u.IsAbs() {
		t.Fatal("Relative URL returned from blob upload with non-relative configuration")
	}
}

func TestBlobDeleteDisabled(t *testing.T) {
	deleteEnabled := false
	env := newTestEnv(t, deleteEnabled)
	defer env.Shutdown()
	args := makeBlobArgs(t)

	imageName := args.imageName
	layerDigest := args.layerDigest
	ref, _ := reference.WithDigest(imageName, layerDigest)
	layerURL, err := env.builder.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("error building url: %v", err)
	}

	resp, err := httpDelete(layerURL)
	if err != nil {
		t.Fatalf("unexpected error deleting when disabled: %v", err)
	}

	checkResponse(t, "status of disabled delete", resp, http.StatusMethodNotAllowed)
}

func testBlobAPI(t *testing.T, env *testEnv, args blobArgs) *testEnv {
	// TODO(stevvooe): This test code is complete junk but it should cover the
	// complete flow. This must be broken down and checked against the
	// specification *before* we submit the final to docker core.
	imageName := args.imageName
	layerFile := args.layerFile
	layerDigest := args.layerDigest

	// -----------------------------------
	// Test fetch for non-existent content
	ref, _ := reference.WithDigest(imageName, layerDigest)
	layerURL, err := env.builder.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("error building url: %v", err)
	}

	resp, err := http.Get(layerURL)
	if err != nil {
		t.Fatalf("unexpected error fetching non-existent layer: %v", err)
	}

	checkResponse(t, "fetching non-existent content", resp, http.StatusNotFound)

	// ------------------------------------------
	// Test head request for non-existent content
	resp, err = http.Head(layerURL)
	if err != nil {
		t.Fatalf("unexpected error checking head on non-existent layer: %v", err)
	}

	checkResponse(t, "checking head on non-existent layer", resp, http.StatusNotFound)

	// ------------------------------------------
	// Start an upload, check the status then cancel
	uploadURLBase, uploadUUID := startPushLayer(t, env, imageName)

	// A status check should work
	resp, err = http.Get(uploadURLBase)
	if err != nil {
		t.Fatalf("unexpected error getting upload status: %v", err)
	}
	checkResponse(t, "status of deleted upload", resp, http.StatusNoContent)
	checkHeaders(t, resp, http.Header{
		"Location":           []string{"*"},
		"Range":              []string{"0-0"},
		"Docker-Upload-UUID": []string{uploadUUID},
	})

	req, err := http.NewRequest("DELETE", uploadURLBase, nil)
	if err != nil {
		t.Fatalf("unexpected error creating delete request: %v", err)
	}

	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected error sending delete request: %v", err)
	}

	checkResponse(t, "deleting upload", resp, http.StatusNoContent)

	// A status check should result in 404
	resp, err = http.Get(uploadURLBase)
	if err != nil {
		t.Fatalf("unexpected error getting upload status: %v", err)
	}
	checkResponse(t, "status of deleted upload", resp, http.StatusNotFound)

	// -----------------------------------------
	// Do layer push with an empty body and different digest
	uploadURLBase, uploadUUID = startPushLayer(t, env, imageName)
	resp, err = doPushLayer(t, env.builder, imageName, layerDigest, uploadURLBase, bytes.NewReader([]byte{}))
	if err != nil {
		t.Fatalf("unexpected error doing bad layer push: %v", err)
	}

	checkResponse(t, "bad layer push", resp, http.StatusBadRequest)
	checkBodyHasErrorCodes(t, "bad layer push", resp, v2.ErrorCodeDigestInvalid)

	// -----------------------------------------
	// Do layer push with an empty body and correct digest
	zeroDigest, err := digest.FromReader(bytes.NewReader([]byte{}))
	if err != nil {
		t.Fatalf("unexpected error digesting empty buffer: %v", err)
	}

	uploadURLBase, uploadUUID = startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, zeroDigest, uploadURLBase, bytes.NewReader([]byte{}))

	// -----------------------------------------
	// Do layer push with an empty body and correct digest

	// This is a valid but empty tarfile!
	emptyTar := bytes.Repeat([]byte("\x00"), 1024)
	emptyDigest, err := digest.FromReader(bytes.NewReader(emptyTar))
	if err != nil {
		t.Fatalf("unexpected error digesting empty tar: %v", err)
	}

	uploadURLBase, uploadUUID = startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, emptyDigest, uploadURLBase, bytes.NewReader(emptyTar))

	// ------------------------------------------
	// Now, actually do successful upload.
	layerLength, _ := layerFile.Seek(0, os.SEEK_END)
	layerFile.Seek(0, os.SEEK_SET)

	uploadURLBase, uploadUUID = startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, layerDigest, uploadURLBase, layerFile)

	// ------------------------------------------
	// Now, push just a chunk
	layerFile.Seek(0, 0)

	canonicalDigester := digest.Canonical.Digester()
	if _, err := io.Copy(canonicalDigester.Hash(), layerFile); err != nil {
		t.Fatalf("error copying to digest: %v", err)
	}
	canonicalDigest := canonicalDigester.Digest()

	layerFile.Seek(0, 0)
	uploadURLBase, uploadUUID = startPushLayer(t, env, imageName)
	uploadURLBase, dgst := pushChunk(t, env.builder, imageName, uploadURLBase, layerFile, layerLength)
	finishUpload(t, env.builder, imageName, uploadURLBase, dgst)

	// ------------------------
	// Use a head request to see if the layer exists.
	resp, err = http.Head(layerURL)
	if err != nil {
		t.Fatalf("unexpected error checking head on existing layer: %v", err)
	}

	checkResponse(t, "checking head on existing layer", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Length":        []string{fmt.Sprint(layerLength)},
		"Docker-Content-Digest": []string{canonicalDigest.String()},
	})

	// ----------------
	// Fetch the layer!
	resp, err = http.Get(layerURL)
	if err != nil {
		t.Fatalf("unexpected error fetching layer: %v", err)
	}

	checkResponse(t, "fetching layer", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Length":        []string{fmt.Sprint(layerLength)},
		"Docker-Content-Digest": []string{canonicalDigest.String()},
	})

	// Verify the body
	verifier := layerDigest.Verifier()
	io.Copy(verifier, resp.Body)

	if !verifier.Verified() {
		t.Fatalf("response body did not pass verification")
	}

	// ----------------
	// Fetch the layer with an invalid digest
	badURL := strings.Replace(layerURL, "sha256", "sha257", 1)
	resp, err = http.Get(badURL)
	if err != nil {
		t.Fatalf("unexpected error fetching layer: %v", err)
	}

	checkResponse(t, "fetching layer bad digest", resp, http.StatusBadRequest)

	// Cache headers
	resp, err = http.Get(layerURL)
	if err != nil {
		t.Fatalf("unexpected error fetching layer: %v", err)
	}

	checkResponse(t, "fetching layer", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Length":        []string{fmt.Sprint(layerLength)},
		"Docker-Content-Digest": []string{canonicalDigest.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, canonicalDigest)},
		"Cache-Control":         []string{"max-age=31536000"},
	})

	// Matching etag, gives 304
	etag := resp.Header.Get("Etag")
	req, err = http.NewRequest("GET", layerURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)

	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching layer with etag", resp, http.StatusNotModified)

	// Non-matching etag, gives 200
	req, err = http.NewRequest("GET", layerURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", "")
	resp, err = http.DefaultClient.Do(req)
	checkResponse(t, "fetching layer with invalid etag", resp, http.StatusOK)

	// Missing tests:
	// 	- Upload the same tar file under and different repository and
	//       ensure the content remains uncorrupted.
	return env
}

func testBlobDelete(t *testing.T, env *testEnv, args blobArgs) {
	// Upload a layer
	imageName := args.imageName
	layerFile := args.layerFile
	layerDigest := args.layerDigest

	ref, _ := reference.WithDigest(imageName, layerDigest)
	layerURL, err := env.builder.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf(err.Error())
	}
	// ---------------
	// Delete a layer
	resp, err := httpDelete(layerURL)
	if err != nil {
		t.Fatalf("unexpected error deleting layer: %v", err)
	}

	checkResponse(t, "deleting layer", resp, http.StatusAccepted)
	checkHeaders(t, resp, http.Header{
		"Content-Length": []string{"0"},
	})

	// ---------------
	// Try and get it back
	// Use a head request to see if the layer exists.
	resp, err = http.Head(layerURL)
	if err != nil {
		t.Fatalf("unexpected error checking head on existing layer: %v", err)
	}

	checkResponse(t, "checking existence of deleted layer", resp, http.StatusNotFound)

	// Delete already deleted layer
	resp, err = httpDelete(layerURL)
	if err != nil {
		t.Fatalf("unexpected error deleting layer: %v", err)
	}

	checkResponse(t, "deleting layer", resp, http.StatusNotFound)

	// ----------------
	// Attempt to delete a layer with an invalid digest
	badURL := strings.Replace(layerURL, "sha256", "sha257", 1)
	resp, err = httpDelete(badURL)
	if err != nil {
		t.Fatalf("unexpected error fetching layer: %v", err)
	}

	checkResponse(t, "deleting layer bad digest", resp, http.StatusBadRequest)

	// ----------------
	// Reupload previously deleted blob
	layerFile.Seek(0, os.SEEK_SET)

	uploadURLBase, _ := startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, layerDigest, uploadURLBase, layerFile)

	layerFile.Seek(0, os.SEEK_SET)
	canonicalDigester := digest.Canonical.Digester()
	if _, err := io.Copy(canonicalDigester.Hash(), layerFile); err != nil {
		t.Fatalf("error copying to digest: %v", err)
	}
	canonicalDigest := canonicalDigester.Digest()

	// ------------------------
	// Use a head request to see if it exists
	resp, err = http.Head(layerURL)
	if err != nil {
		t.Fatalf("unexpected error checking head on existing layer: %v", err)
	}

	layerLength, _ := layerFile.Seek(0, os.SEEK_END)
	checkResponse(t, "checking head on reuploaded layer", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Content-Length":        []string{fmt.Sprint(layerLength)},
		"Docker-Content-Digest": []string{canonicalDigest.String()},
	})
}

func TestDeleteDisabled(t *testing.T) {
	env := newTestEnv(t, false)
	defer env.Shutdown()

	imageName, _ := reference.WithName("foo/bar")
	// "build" our layer file
	layerFile, layerDigest, err := testutil.CreateRandomTarFile()
	if err != nil {
		t.Fatalf("error creating random layer file: %v", err)
	}

	ref, _ := reference.WithDigest(imageName, layerDigest)
	layerURL, err := env.builder.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("Error building blob URL")
	}
	uploadURLBase, _ := startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, layerDigest, uploadURLBase, layerFile)

	resp, err := httpDelete(layerURL)
	if err != nil {
		t.Fatalf("unexpected error deleting layer: %v", err)
	}

	checkResponse(t, "deleting layer with delete disabled", resp, http.StatusMethodNotAllowed)
}

func TestDeleteReadOnly(t *testing.T) {
	env := newTestEnv(t, true)
	defer env.Shutdown()

	imageName, _ := reference.WithName("foo/bar")
	// "build" our layer file
	layerFile, layerDigest, err := testutil.CreateRandomTarFile()
	if err != nil {
		t.Fatalf("error creating random layer file: %v", err)
	}

	ref, _ := reference.WithDigest(imageName, layerDigest)
	layerURL, err := env.builder.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("Error building blob URL")
	}
	uploadURLBase, _ := startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, layerDigest, uploadURLBase, layerFile)

	env.app.readOnly = true

	resp, err := httpDelete(layerURL)
	if err != nil {
		t.Fatalf("unexpected error deleting layer: %v", err)
	}

	checkResponse(t, "deleting layer in read-only mode", resp, http.StatusMethodNotAllowed)
}

func TestStartPushReadOnly(t *testing.T) {
	env := newTestEnv(t, true)
	defer env.Shutdown()
	env.app.readOnly = true

	imageName, _ := reference.WithName("foo/bar")

	layerUploadURL, err := env.builder.BuildBlobUploadURL(imageName)
	if err != nil {
		t.Fatalf("unexpected error building layer upload url: %v", err)
	}

	resp, err := http.Post(layerUploadURL, "", nil)
	if err != nil {
		t.Fatalf("unexpected error starting layer push: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "starting push in read-only mode", resp, http.StatusMethodNotAllowed)
}

func httpDelete(url string) (*http.Response, error) {
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	//	defer resp.Body.Close()
	return resp, err
}

type manifestArgs struct {
	imageName reference.Named
	mediaType string
	manifest  distribution.Manifest
	dgst      digest.Digest
}

func TestManifestAPI(t *testing.T) {
	schema1Repo, _ := reference.WithName("foo/schema1")
	schema2Repo, _ := reference.WithName("foo/schema2")

	deleteEnabled := false
	env1 := newTestEnv(t, deleteEnabled)
	defer env1.Shutdown()
	testManifestAPISchema1(t, env1, schema1Repo)
	schema2Args := testManifestAPISchema2(t, env1, schema2Repo)
	testManifestAPIManifestList(t, env1, schema2Args)

	deleteEnabled = true
	env2 := newTestEnv(t, deleteEnabled)
	defer env2.Shutdown()
	testManifestAPISchema1(t, env2, schema1Repo)
	schema2Args = testManifestAPISchema2(t, env2, schema2Repo)
	testManifestAPIManifestList(t, env2, schema2Args)
}

// storageManifestErrDriverFactory implements the factory.StorageDriverFactory interface.
type storageManifestErrDriverFactory struct{}

const (
	repositoryWithManifestNotFound    = "manifesttagnotfound"
	repositoryWithManifestInvalidPath = "manifestinvalidpath"
	repositoryWithManifestBadLink     = "manifestbadlink"
	repositoryWithGenericStorageError = "genericstorageerr"
)

func (factory *storageManifestErrDriverFactory) Create(parameters map[string]interface{}) (storagedriver.StorageDriver, error) {
	// Initialize the mock driver
	var errGenericStorage = errors.New("generic storage error")
	return &mockErrorDriver{
		returnErrs: []mockErrorMapping{
			{
				pathMatch: fmt.Sprintf("%s/_manifests/tags", repositoryWithManifestNotFound),
				content:   nil,
				err:       storagedriver.PathNotFoundError{},
			},
			{
				pathMatch: fmt.Sprintf("%s/_manifests/tags", repositoryWithManifestInvalidPath),
				content:   nil,
				err:       storagedriver.InvalidPathError{},
			},
			{
				pathMatch: fmt.Sprintf("%s/_manifests/tags", repositoryWithManifestBadLink),
				content:   []byte("this is a bad sha"),
				err:       nil,
			},
			{
				pathMatch: fmt.Sprintf("%s/_manifests/tags", repositoryWithGenericStorageError),
				content:   nil,
				err:       errGenericStorage,
			},
		},
	}, nil
}

type mockErrorMapping struct {
	pathMatch string
	content   []byte
	err       error
}

// mockErrorDriver implements StorageDriver to force storage error on manifest request
type mockErrorDriver struct {
	storagedriver.StorageDriver
	returnErrs []mockErrorMapping
}

func (dr *mockErrorDriver) GetContent(ctx context.Context, path string) ([]byte, error) {
	for _, returns := range dr.returnErrs {
		if strings.Contains(path, returns.pathMatch) {
			return returns.content, returns.err
		}
	}
	return nil, errors.New("Unknown storage error")
}

func TestGetManifestWithStorageError(t *testing.T) {
	factory.Register("storagemanifesterror", &storageManifestErrDriverFactory{})
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"storagemanifesterror": configuration.Parameters{},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
	}
	config.HTTP.Headers = headerConfig
	env1 := newTestEnvWithConfig(t, &config)
	defer env1.Shutdown()

	repo, _ := reference.WithName(repositoryWithManifestNotFound)
	testManifestWithStorageError(t, env1, repo, http.StatusNotFound, v2.ErrorCodeManifestUnknown)

	repo, _ = reference.WithName(repositoryWithGenericStorageError)
	testManifestWithStorageError(t, env1, repo, http.StatusInternalServerError, errcode.ErrorCodeUnknown)

	repo, _ = reference.WithName(repositoryWithManifestInvalidPath)
	testManifestWithStorageError(t, env1, repo, http.StatusInternalServerError, errcode.ErrorCodeUnknown)

	repo, _ = reference.WithName(repositoryWithManifestBadLink)
	testManifestWithStorageError(t, env1, repo, http.StatusInternalServerError, errcode.ErrorCodeUnknown)
}

func TestManifestDelete(t *testing.T) {
	schema1Repo, _ := reference.WithName("foo/schema1")
	schema2Repo, _ := reference.WithName("foo/schema2")

	deleteEnabled := true
	env := newTestEnv(t, deleteEnabled)
	defer env.Shutdown()
	schema1Args := testManifestAPISchema1(t, env, schema1Repo)
	testManifestDelete(t, env, schema1Args)
	schema2Args := testManifestAPISchema2(t, env, schema2Repo)
	testManifestDelete(t, env, schema2Args)
}

func TestManifestDeleteDisabled(t *testing.T) {
	schema1Repo, _ := reference.WithName("foo/schema1")
	deleteEnabled := false
	env := newTestEnv(t, deleteEnabled)
	defer env.Shutdown()
	testManifestDeleteDisabled(t, env, schema1Repo)
}

func testManifestDeleteDisabled(t *testing.T, env *testEnv, imageName reference.Named) {
	ref, _ := reference.WithDigest(imageName, digestSha256EmptyTar)
	manifestURL, err := env.builder.BuildManifestURL(ref)
	if err != nil {
		t.Fatalf("unexpected error getting manifest url: %v", err)
	}

	resp, err := httpDelete(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error deleting manifest %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "status of disabled delete of manifest", resp, http.StatusMethodNotAllowed)
}

func testManifestWithStorageError(t *testing.T, env *testEnv, imageName reference.Named, expectedStatusCode int, expectedErrorCode errcode.ErrorCode) {
	tag := "latest"
	tagRef, _ := reference.WithTag(imageName, tag)
	manifestURL, err := env.builder.BuildManifestURL(tagRef)
	if err != nil {
		t.Fatalf("unexpected error getting manifest url: %v", err)
	}

	// -----------------------------
	// Attempt to fetch the manifest
	resp, err := http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error getting manifest: %v", err)
	}
	defer resp.Body.Close()
	checkResponse(t, "getting non-existent manifest", resp, expectedStatusCode)
	checkBodyHasErrorCodes(t, "getting non-existent manifest", resp, expectedErrorCode)
	return
}

func testManifestAPISchema1(t *testing.T, env *testEnv, imageName reference.Named) manifestArgs {
	tag := "thetag"
	args := manifestArgs{imageName: imageName}

	tagRef, _ := reference.WithTag(imageName, tag)
	manifestURL, err := env.builder.BuildManifestURL(tagRef)
	if err != nil {
		t.Fatalf("unexpected error getting manifest url: %v", err)
	}

	// -----------------------------
	// Attempt to fetch the manifest
	resp, err := http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error getting manifest: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "getting non-existent manifest", resp, http.StatusNotFound)
	checkBodyHasErrorCodes(t, "getting non-existent manifest", resp, v2.ErrorCodeManifestUnknown)

	tagsURL, err := env.builder.BuildTagsURL(imageName)
	if err != nil {
		t.Fatalf("unexpected error building tags url: %v", err)
	}

	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	// Check that we get an unknown repository error when asking for tags
	checkResponse(t, "getting unknown manifest tags", resp, http.StatusNotFound)
	checkBodyHasErrorCodes(t, "getting unknown manifest tags", resp, v2.ErrorCodeNameUnknown)

	// --------------------------------
	// Attempt to push unsigned manifest with missing layers
	unsignedManifest := &schema1.Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name: imageName.Name(),
		Tag:  tag,
		FSLayers: []schema1.FSLayer{
			{
				BlobSum: "asdf",
			},
			{
				BlobSum: "qwer",
			},
		},
		History: []schema1.History{
			{
				V1Compatibility: "",
			},
			{
				V1Compatibility: "",
			},
		},
	}

	resp = putManifest(t, "putting unsigned manifest", manifestURL, "", unsignedManifest)
	defer resp.Body.Close()
	checkResponse(t, "putting unsigned manifest", resp, http.StatusBadRequest)
	_, p, counts := checkBodyHasErrorCodes(t, "putting unsigned manifest", resp, v2.ErrorCodeManifestInvalid)

	expectedCounts := map[errcode.ErrorCode]int{
		v2.ErrorCodeManifestInvalid: 1,
	}

	if !reflect.DeepEqual(counts, expectedCounts) {
		t.Fatalf("unexpected number of error codes encountered: %v\n!=\n%v\n---\n%s", counts, expectedCounts, string(p))
	}

	// sign the manifest and still get some interesting errors.
	sm, err := schema1.Sign(unsignedManifest, env.pk)
	if err != nil {
		t.Fatalf("error signing manifest: %v", err)
	}

	resp = putManifest(t, "putting signed manifest with errors", manifestURL, "", sm)
	defer resp.Body.Close()
	checkResponse(t, "putting signed manifest with errors", resp, http.StatusBadRequest)
	_, p, counts = checkBodyHasErrorCodes(t, "putting signed manifest with errors", resp,
		v2.ErrorCodeManifestBlobUnknown, v2.ErrorCodeDigestInvalid)

	expectedCounts = map[errcode.ErrorCode]int{
		v2.ErrorCodeManifestBlobUnknown: 2,
		v2.ErrorCodeDigestInvalid:       2,
	}

	if !reflect.DeepEqual(counts, expectedCounts) {
		t.Fatalf("unexpected number of error codes encountered: %v\n!=\n%v\n---\n%s", counts, expectedCounts, string(p))
	}

	// TODO(stevvooe): Add a test case where we take a mostly valid registry,
	// tamper with the content and ensure that we get an unverified manifest
	// error.

	// Push 2 random layers
	expectedLayers := make(map[digest.Digest]io.ReadSeeker)

	for i := range unsignedManifest.FSLayers {
		rs, dgstStr, err := testutil.CreateRandomTarFile()

		if err != nil {
			t.Fatalf("error creating random layer %d: %v", i, err)
		}
		dgst := digest.Digest(dgstStr)

		expectedLayers[dgst] = rs
		unsignedManifest.FSLayers[i].BlobSum = dgst

		uploadURLBase, _ := startPushLayer(t, env, imageName)
		pushLayer(t, env.builder, imageName, dgst, uploadURLBase, rs)
	}

	// -------------------
	// Push the signed manifest with all layers pushed.
	signedManifest, err := schema1.Sign(unsignedManifest, env.pk)
	if err != nil {
		t.Fatalf("unexpected error signing manifest: %v", err)
	}

	dgst := digest.FromBytes(signedManifest.Canonical)
	args.manifest = signedManifest
	args.dgst = dgst

	digestRef, _ := reference.WithDigest(imageName, dgst)
	manifestDigestURL, err := env.builder.BuildManifestURL(digestRef)
	checkErr(t, err, "building manifest url")

	resp = putManifest(t, "putting signed manifest no error", manifestURL, "", signedManifest)
	checkResponse(t, "putting signed manifest no error", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// --------------------
	// Push by digest -- should get same result
	resp = putManifest(t, "putting signed manifest", manifestDigestURL, "", signedManifest)
	checkResponse(t, "putting signed manifest", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// ------------------
	// Fetch by tag name
	resp, err = http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error fetching manifest: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifest schema1.SignedManifest
	dec := json.NewDecoder(resp.Body)

	if err := dec.Decode(&fetchedManifest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	if !bytes.Equal(fetchedManifest.Canonical, signedManifest.Canonical) {
		t.Fatalf("manifests do not match")
	}

	// ---------------
	// Fetch by digest
	resp, err = http.Get(manifestDigestURL)
	checkErr(t, err, "fetching manifest by digest")
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifestByDigest schema1.SignedManifest
	dec = json.NewDecoder(resp.Body)
	if err := dec.Decode(&fetchedManifestByDigest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	if !bytes.Equal(fetchedManifestByDigest.Canonical, signedManifest.Canonical) {
		t.Fatalf("manifests do not match")
	}

	// check signature was roundtripped
	signatures, err := fetchedManifestByDigest.Signatures()
	if err != nil {
		t.Fatal(err)
	}

	if len(signatures) != 1 {
		t.Fatalf("expected 1 signature from manifest, got: %d", len(signatures))
	}

	// Re-sign, push and pull the same digest
	sm2, err := schema1.Sign(&fetchedManifestByDigest.Manifest, env.pk)
	if err != nil {
		t.Fatal(err)

	}

	// Re-push with a few different Content-Types. The official schema1
	// content type should work, as should application/json with/without a
	// charset.
	resp = putManifest(t, "re-putting signed manifest", manifestDigestURL, schema1.MediaTypeSignedManifest, sm2)
	checkResponse(t, "re-putting signed manifest", resp, http.StatusCreated)
	resp = putManifest(t, "re-putting signed manifest", manifestDigestURL, "application/json; charset=utf-8", sm2)
	checkResponse(t, "re-putting signed manifest", resp, http.StatusCreated)
	resp = putManifest(t, "re-putting signed manifest", manifestDigestURL, "application/json", sm2)
	checkResponse(t, "re-putting signed manifest", resp, http.StatusCreated)

	resp, err = http.Get(manifestDigestURL)
	checkErr(t, err, "re-fetching manifest by digest")
	defer resp.Body.Close()

	checkResponse(t, "re-fetching uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	dec = json.NewDecoder(resp.Body)
	if err := dec.Decode(&fetchedManifestByDigest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	// check only 1 signature is returned
	signatures, err = fetchedManifestByDigest.Signatures()
	if err != nil {
		t.Fatal(err)
	}

	if len(signatures) != 1 {
		t.Fatalf("expected 2 signature from manifest, got: %d", len(signatures))
	}

	// Get by name with etag, gives 304
	etag := resp.Header.Get("Etag")
	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by name with etag", resp, http.StatusNotModified)

	// Get by digest with etag, gives 304
	req, err = http.NewRequest("GET", manifestDigestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by dgst with etag", resp, http.StatusNotModified)

	// Ensure that the tag is listed.
	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "getting tags", resp, http.StatusOK)
	dec = json.NewDecoder(resp.Body)

	var tagsResponse tagsAPIResponse

	if err := dec.Decode(&tagsResponse); err != nil {
		t.Fatalf("unexpected error decoding error response: %v", err)
	}

	if tagsResponse.Name != imageName.Name() {
		t.Fatalf("tags name should match image name: %v != %v", tagsResponse.Name, imageName.Name())
	}

	if len(tagsResponse.Tags) != 1 {
		t.Fatalf("expected some tags in response: %v", tagsResponse.Tags)
	}

	if tagsResponse.Tags[0] != tag {
		t.Fatalf("tag not as expected: %q != %q", tagsResponse.Tags[0], tag)
	}

	// Attempt to put a manifest with mismatching FSLayer and History array cardinalities

	unsignedManifest.History = append(unsignedManifest.History, schema1.History{
		V1Compatibility: "",
	})
	invalidSigned, err := schema1.Sign(unsignedManifest, env.pk)
	if err != nil {
		t.Fatalf("error signing manifest")
	}

	resp = putManifest(t, "putting invalid signed manifest", manifestDigestURL, "", invalidSigned)
	checkResponse(t, "putting invalid signed manifest", resp, http.StatusBadRequest)

	return args
}

func testManifestAPISchema2(t *testing.T, env *testEnv, imageName reference.Named) manifestArgs {
	tag := "schema2tag"
	args := manifestArgs{
		imageName: imageName,
		mediaType: schema2.MediaTypeManifest,
	}

	tagRef, _ := reference.WithTag(imageName, tag)
	manifestURL, err := env.builder.BuildManifestURL(tagRef)
	if err != nil {
		t.Fatalf("unexpected error getting manifest url: %v", err)
	}

	// -----------------------------
	// Attempt to fetch the manifest
	resp, err := http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error getting manifest: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "getting non-existent manifest", resp, http.StatusNotFound)
	checkBodyHasErrorCodes(t, "getting non-existent manifest", resp, v2.ErrorCodeManifestUnknown)

	tagsURL, err := env.builder.BuildTagsURL(imageName)
	if err != nil {
		t.Fatalf("unexpected error building tags url: %v", err)
	}

	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	// Check that we get an unknown repository error when asking for tags
	checkResponse(t, "getting unknown manifest tags", resp, http.StatusNotFound)
	checkBodyHasErrorCodes(t, "getting unknown manifest tags", resp, v2.ErrorCodeNameUnknown)

	// --------------------------------
	// Attempt to push manifest with missing config and missing layers
	manifest := &schema2.Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 2,
			MediaType:     schema2.MediaTypeManifest,
		},
		Config: distribution.Descriptor{
			Digest:    "sha256:1a9ec845ee94c202b2d5da74a24f0ed2058318bfa9879fa541efaecba272e86b",
			Size:      3253,
			MediaType: schema2.MediaTypeImageConfig,
		},
		Layers: []distribution.Descriptor{
			{
				Digest:    "sha256:463434349086340864309863409683460843608348608934092322395278926a",
				Size:      6323,
				MediaType: schema2.MediaTypeLayer,
			},
			{
				Digest:    "sha256:630923423623623423352523525237238023652897356239852383652aaaaaaa",
				Size:      6863,
				MediaType: schema2.MediaTypeLayer,
			},
		},
	}

	resp = putManifest(t, "putting missing config manifest", manifestURL, schema2.MediaTypeManifest, manifest)
	defer resp.Body.Close()
	checkResponse(t, "putting missing config manifest", resp, http.StatusBadRequest)
	_, p, counts := checkBodyHasErrorCodes(t, "putting missing config manifest", resp, v2.ErrorCodeManifestBlobUnknown)

	expectedCounts := map[errcode.ErrorCode]int{
		v2.ErrorCodeManifestBlobUnknown: 3,
	}

	if !reflect.DeepEqual(counts, expectedCounts) {
		t.Fatalf("unexpected number of error codes encountered: %v\n!=\n%v\n---\n%s", counts, expectedCounts, string(p))
	}

	// Push a config, and reference it in the manifest
	sampleConfig := []byte(`{
		"architecture": "amd64",
		"history": [
		  {
		    "created": "2015-10-31T22:22:54.690851953Z",
		    "created_by": "/bin/sh -c #(nop) ADD file:a3bc1e842b69636f9df5256c49c5374fb4eef1e281fe3f282c65fb853ee171c5 in /"
		  },
		  {
		    "created": "2015-10-31T22:22:55.613815829Z",
		    "created_by": "/bin/sh -c #(nop) CMD [\"sh\"]"
		  }
		],
		"rootfs": {
		  "diff_ids": [
		    "sha256:c6f988f4874bb0add23a778f753c65efe992244e148a1d2ec2a8b664fb66bbd1",
		    "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
		  ],
		  "type": "layers"
		}
	}`)
	sampleConfigDigest := digest.FromBytes(sampleConfig)

	uploadURLBase, _ := startPushLayer(t, env, imageName)
	pushLayer(t, env.builder, imageName, sampleConfigDigest, uploadURLBase, bytes.NewReader(sampleConfig))
	manifest.Config.Digest = sampleConfigDigest
	manifest.Config.Size = int64(len(sampleConfig))

	// The manifest should still be invalid, because its layer doesn't exist
	resp = putManifest(t, "putting missing layer manifest", manifestURL, schema2.MediaTypeManifest, manifest)
	defer resp.Body.Close()
	checkResponse(t, "putting missing layer manifest", resp, http.StatusBadRequest)
	_, p, counts = checkBodyHasErrorCodes(t, "getting unknown manifest tags", resp, v2.ErrorCodeManifestBlobUnknown)

	expectedCounts = map[errcode.ErrorCode]int{
		v2.ErrorCodeManifestBlobUnknown: 2,
	}

	if !reflect.DeepEqual(counts, expectedCounts) {
		t.Fatalf("unexpected number of error codes encountered: %v\n!=\n%v\n---\n%s", counts, expectedCounts, string(p))
	}

	// Push 2 random layers
	expectedLayers := make(map[digest.Digest]io.ReadSeeker)

	for i := range manifest.Layers {
		rs, dgstStr, err := testutil.CreateRandomTarFile()

		if err != nil {
			t.Fatalf("error creating random layer %d: %v", i, err)
		}
		dgst := digest.Digest(dgstStr)

		expectedLayers[dgst] = rs
		manifest.Layers[i].Digest = dgst

		uploadURLBase, _ := startPushLayer(t, env, imageName)
		pushLayer(t, env.builder, imageName, dgst, uploadURLBase, rs)
	}

	// -------------------
	// Push the manifest with all layers pushed.
	deserializedManifest, err := schema2.FromStruct(*manifest)
	if err != nil {
		t.Fatalf("could not create DeserializedManifest: %v", err)
	}
	_, canonical, err := deserializedManifest.Payload()
	if err != nil {
		t.Fatalf("could not get manifest payload: %v", err)
	}
	dgst := digest.FromBytes(canonical)
	args.dgst = dgst
	args.manifest = deserializedManifest

	digestRef, _ := reference.WithDigest(imageName, dgst)
	manifestDigestURL, err := env.builder.BuildManifestURL(digestRef)
	checkErr(t, err, "building manifest url")

	resp = putManifest(t, "putting manifest no error", manifestURL, schema2.MediaTypeManifest, manifest)
	checkResponse(t, "putting manifest no error", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// --------------------
	// Push by digest -- should get same result
	resp = putManifest(t, "putting manifest by digest", manifestDigestURL, schema2.MediaTypeManifest, manifest)
	checkResponse(t, "putting manifest by digest", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// ------------------
	// Fetch by tag name
	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("Accept", schema2.MediaTypeManifest)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected error fetching manifest: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifest schema2.DeserializedManifest
	dec := json.NewDecoder(resp.Body)

	if err := dec.Decode(&fetchedManifest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	_, fetchedCanonical, err := fetchedManifest.Payload()
	if err != nil {
		t.Fatalf("error getting manifest payload: %v", err)
	}

	if !bytes.Equal(fetchedCanonical, canonical) {
		t.Fatalf("manifests do not match")
	}

	// ---------------
	// Fetch by digest
	req, err = http.NewRequest("GET", manifestDigestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("Accept", schema2.MediaTypeManifest)
	resp, err = http.DefaultClient.Do(req)
	checkErr(t, err, "fetching manifest by digest")
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifestByDigest schema2.DeserializedManifest
	dec = json.NewDecoder(resp.Body)
	if err := dec.Decode(&fetchedManifestByDigest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	_, fetchedCanonical, err = fetchedManifest.Payload()
	if err != nil {
		t.Fatalf("error getting manifest payload: %v", err)
	}

	if !bytes.Equal(fetchedCanonical, canonical) {
		t.Fatalf("manifests do not match")
	}

	// Get by name with etag, gives 304
	etag := resp.Header.Get("Etag")
	req, err = http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by name with etag", resp, http.StatusNotModified)

	// Get by digest with etag, gives 304
	req, err = http.NewRequest("GET", manifestDigestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by dgst with etag", resp, http.StatusNotModified)

	// Ensure that the tag is listed.
	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "getting unknown manifest tags", resp, http.StatusOK)
	dec = json.NewDecoder(resp.Body)

	var tagsResponse tagsAPIResponse

	if err := dec.Decode(&tagsResponse); err != nil {
		t.Fatalf("unexpected error decoding error response: %v", err)
	}

	if tagsResponse.Name != imageName.Name() {
		t.Fatalf("tags name should match image name: %v != %v", tagsResponse.Name, imageName)
	}

	if len(tagsResponse.Tags) != 1 {
		t.Fatalf("expected some tags in response: %v", tagsResponse.Tags)
	}

	if tagsResponse.Tags[0] != tag {
		t.Fatalf("tag not as expected: %q != %q", tagsResponse.Tags[0], tag)
	}

	// ------------------
	// Fetch as a schema1 manifest
	resp, err = http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error fetching manifest as schema1: %v", err)
	}
	defer resp.Body.Close()

	manifestBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("error reading response body: %v", err)
	}

	checkResponse(t, "fetching uploaded manifest as schema1", resp, http.StatusOK)

	m, desc, err := distribution.UnmarshalManifest(schema1.MediaTypeManifest, manifestBytes)
	if err != nil {
		t.Fatalf("unexpected error unmarshalling manifest: %v", err)
	}

	fetchedSchema1Manifest, ok := m.(*schema1.SignedManifest)
	if !ok {
		t.Fatalf("expecting schema1 manifest")
	}

	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{desc.Digest.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, desc.Digest)},
	})

	if fetchedSchema1Manifest.Manifest.SchemaVersion != 1 {
		t.Fatal("wrong schema version")
	}
	if fetchedSchema1Manifest.Architecture != "amd64" {
		t.Fatal("wrong architecture")
	}
	if fetchedSchema1Manifest.Name != imageName.Name() {
		t.Fatal("wrong image name")
	}
	if fetchedSchema1Manifest.Tag != tag {
		t.Fatal("wrong tag")
	}
	if len(fetchedSchema1Manifest.FSLayers) != 2 {
		t.Fatal("wrong number of FSLayers")
	}
	for i := range manifest.Layers {
		if fetchedSchema1Manifest.FSLayers[i].BlobSum != manifest.Layers[len(manifest.Layers)-i-1].Digest {
			t.Fatalf("blob digest mismatch in schema1 manifest for layer %d", i)
		}
	}
	if len(fetchedSchema1Manifest.History) != 2 {
		t.Fatal("wrong number of History entries")
	}

	// Don't check V1Compatibility fields because we're using randomly-generated
	// layers.

	return args
}

func testManifestAPIManifestList(t *testing.T, env *testEnv, args manifestArgs) {
	imageName := args.imageName
	tag := "manifestlisttag"

	tagRef, _ := reference.WithTag(imageName, tag)
	manifestURL, err := env.builder.BuildManifestURL(tagRef)
	if err != nil {
		t.Fatalf("unexpected error getting manifest url: %v", err)
	}

	// --------------------------------
	// Attempt to push manifest list that refers to an unknown manifest
	manifestList := &manifestlist.ManifestList{
		Versioned: manifest.Versioned{
			SchemaVersion: 2,
			MediaType:     manifestlist.MediaTypeManifestList,
		},
		Manifests: []manifestlist.ManifestDescriptor{
			{
				Descriptor: distribution.Descriptor{
					Digest:    "sha256:1a9ec845ee94c202b2d5da74a24f0ed2058318bfa9879fa541efaecba272e86b",
					Size:      3253,
					MediaType: schema2.MediaTypeManifest,
				},
				Platform: manifestlist.PlatformSpec{
					Architecture: "amd64",
					OS:           "linux",
				},
			},
		},
	}

	resp := putManifest(t, "putting missing manifest manifestlist", manifestURL, manifestlist.MediaTypeManifestList, manifestList)
	defer resp.Body.Close()
	checkResponse(t, "putting missing manifest manifestlist", resp, http.StatusBadRequest)
	_, p, counts := checkBodyHasErrorCodes(t, "putting missing manifest manifestlist", resp, v2.ErrorCodeManifestBlobUnknown)

	expectedCounts := map[errcode.ErrorCode]int{
		v2.ErrorCodeManifestBlobUnknown: 1,
	}

	if !reflect.DeepEqual(counts, expectedCounts) {
		t.Fatalf("unexpected number of error codes encountered: %v\n!=\n%v\n---\n%s", counts, expectedCounts, string(p))
	}

	// -------------------
	// Push a manifest list that references an actual manifest
	manifestList.Manifests[0].Digest = args.dgst
	deserializedManifestList, err := manifestlist.FromDescriptors(manifestList.Manifests)
	if err != nil {
		t.Fatalf("could not create DeserializedManifestList: %v", err)
	}
	_, canonical, err := deserializedManifestList.Payload()
	if err != nil {
		t.Fatalf("could not get manifest list payload: %v", err)
	}
	dgst := digest.FromBytes(canonical)

	digestRef, _ := reference.WithDigest(imageName, dgst)
	manifestDigestURL, err := env.builder.BuildManifestURL(digestRef)
	checkErr(t, err, "building manifest url")

	resp = putManifest(t, "putting manifest list no error", manifestURL, manifestlist.MediaTypeManifestList, deserializedManifestList)
	checkResponse(t, "putting manifest list no error", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// --------------------
	// Push by digest -- should get same result
	resp = putManifest(t, "putting manifest list by digest", manifestDigestURL, manifestlist.MediaTypeManifestList, deserializedManifestList)
	checkResponse(t, "putting manifest list by digest", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// ------------------
	// Fetch by tag name
	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	// multiple headers in mixed list format to ensure we parse correctly server-side
	req.Header.Set("Accept", fmt.Sprintf(` %s ; q=0.8 , %s ; q=0.5 `, manifestlist.MediaTypeManifestList, schema1.MediaTypeSignedManifest))
	req.Header.Add("Accept", schema2.MediaTypeManifest)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("unexpected error fetching manifest list: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest list", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifestList manifestlist.DeserializedManifestList
	dec := json.NewDecoder(resp.Body)

	if err := dec.Decode(&fetchedManifestList); err != nil {
		t.Fatalf("error decoding fetched manifest list: %v", err)
	}

	_, fetchedCanonical, err := fetchedManifestList.Payload()
	if err != nil {
		t.Fatalf("error getting manifest list payload: %v", err)
	}

	if !bytes.Equal(fetchedCanonical, canonical) {
		t.Fatalf("manifest lists do not match")
	}

	// ---------------
	// Fetch by digest
	req, err = http.NewRequest("GET", manifestDigestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("Accept", manifestlist.MediaTypeManifestList)
	resp, err = http.DefaultClient.Do(req)
	checkErr(t, err, "fetching manifest list by digest")
	defer resp.Body.Close()

	checkResponse(t, "fetching uploaded manifest list", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, dgst)},
	})

	var fetchedManifestListByDigest manifestlist.DeserializedManifestList
	dec = json.NewDecoder(resp.Body)
	if err := dec.Decode(&fetchedManifestListByDigest); err != nil {
		t.Fatalf("error decoding fetched manifest: %v", err)
	}

	_, fetchedCanonical, err = fetchedManifestListByDigest.Payload()
	if err != nil {
		t.Fatalf("error getting manifest list payload: %v", err)
	}

	if !bytes.Equal(fetchedCanonical, canonical) {
		t.Fatalf("manifests do not match")
	}

	// Get by name with etag, gives 304
	etag := resp.Header.Get("Etag")
	req, err = http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by name with etag", resp, http.StatusNotModified)

	// Get by digest with etag, gives 304
	req, err = http.NewRequest("GET", manifestDigestURL, nil)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}
	req.Header.Set("If-None-Match", etag)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Error constructing request: %s", err)
	}

	checkResponse(t, "fetching manifest by dgst with etag", resp, http.StatusNotModified)

	// ------------------
	// Fetch as a schema1 manifest
	resp, err = http.Get(manifestURL)
	if err != nil {
		t.Fatalf("unexpected error fetching manifest list as schema1: %v", err)
	}
	defer resp.Body.Close()

	manifestBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("error reading response body: %v", err)
	}

	checkResponse(t, "fetching uploaded manifest list as schema1", resp, http.StatusOK)

	m, desc, err := distribution.UnmarshalManifest(schema1.MediaTypeManifest, manifestBytes)
	if err != nil {
		t.Fatalf("unexpected error unmarshalling manifest: %v", err)
	}

	fetchedSchema1Manifest, ok := m.(*schema1.SignedManifest)
	if !ok {
		t.Fatalf("expecting schema1 manifest")
	}

	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{desc.Digest.String()},
		"ETag":                  []string{fmt.Sprintf(`"%s"`, desc.Digest)},
	})

	if fetchedSchema1Manifest.Manifest.SchemaVersion != 1 {
		t.Fatal("wrong schema version")
	}
	if fetchedSchema1Manifest.Architecture != "amd64" {
		t.Fatal("wrong architecture")
	}
	if fetchedSchema1Manifest.Name != imageName.Name() {
		t.Fatal("wrong image name")
	}
	if fetchedSchema1Manifest.Tag != tag {
		t.Fatal("wrong tag")
	}
	if len(fetchedSchema1Manifest.FSLayers) != 2 {
		t.Fatal("wrong number of FSLayers")
	}
	layers := args.manifest.(*schema2.DeserializedManifest).Layers
	for i := range layers {
		if fetchedSchema1Manifest.FSLayers[i].BlobSum != layers[len(layers)-i-1].Digest {
			t.Fatalf("blob digest mismatch in schema1 manifest for layer %d", i)
		}
	}
	if len(fetchedSchema1Manifest.History) != 2 {
		t.Fatal("wrong number of History entries")
	}

	// Don't check V1Compatibility fields because we're using randomly-generated
	// layers.
}

func testManifestDelete(t *testing.T, env *testEnv, args manifestArgs) {
	imageName := args.imageName
	dgst := args.dgst
	manifest := args.manifest

	ref, _ := reference.WithDigest(imageName, dgst)
	manifestDigestURL, err := env.builder.BuildManifestURL(ref)
	// ---------------
	// Delete by digest
	resp, err := httpDelete(manifestDigestURL)
	checkErr(t, err, "deleting manifest by digest")

	checkResponse(t, "deleting manifest", resp, http.StatusAccepted)
	checkHeaders(t, resp, http.Header{
		"Content-Length": []string{"0"},
	})

	// ---------------
	// Attempt to fetch deleted manifest
	resp, err = http.Get(manifestDigestURL)
	checkErr(t, err, "fetching deleted manifest by digest")
	defer resp.Body.Close()

	checkResponse(t, "fetching deleted manifest", resp, http.StatusNotFound)

	// ---------------
	// Delete already deleted manifest by digest
	resp, err = httpDelete(manifestDigestURL)
	checkErr(t, err, "re-deleting manifest by digest")

	checkResponse(t, "re-deleting manifest", resp, http.StatusNotFound)

	// --------------------
	// Re-upload manifest by digest
	resp = putManifest(t, "putting manifest", manifestDigestURL, args.mediaType, manifest)
	checkResponse(t, "putting manifest", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// ---------------
	// Attempt to fetch re-uploaded deleted digest
	resp, err = http.Get(manifestDigestURL)
	checkErr(t, err, "fetching re-uploaded manifest by digest")
	defer resp.Body.Close()

	checkResponse(t, "fetching re-uploaded manifest", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// ---------------
	// Attempt to delete an unknown manifest
	unknownDigest := digest.Digest("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	unknownRef, _ := reference.WithDigest(imageName, unknownDigest)
	unknownManifestDigestURL, err := env.builder.BuildManifestURL(unknownRef)
	checkErr(t, err, "building unknown manifest url")

	resp, err = httpDelete(unknownManifestDigestURL)
	checkErr(t, err, "delting unknown manifest by digest")
	checkResponse(t, "fetching deleted manifest", resp, http.StatusNotFound)

	// --------------------
	// Upload manifest by tag
	tag := "atag"
	tagRef, _ := reference.WithTag(imageName, tag)
	manifestTagURL, err := env.builder.BuildManifestURL(tagRef)
	resp = putManifest(t, "putting manifest by tag", manifestTagURL, args.mediaType, manifest)
	checkResponse(t, "putting manifest by tag", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{manifestDigestURL},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	tagsURL, err := env.builder.BuildTagsURL(imageName)
	if err != nil {
		t.Fatalf("unexpected error building tags url: %v", err)
	}

	// Ensure that the tag is listed.
	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	dec := json.NewDecoder(resp.Body)
	var tagsResponse tagsAPIResponse
	if err := dec.Decode(&tagsResponse); err != nil {
		t.Fatalf("unexpected error decoding error response: %v", err)
	}

	if tagsResponse.Name != imageName.Name() {
		t.Fatalf("tags name should match image name: %v != %v", tagsResponse.Name, imageName)
	}

	if len(tagsResponse.Tags) != 1 {
		t.Fatalf("expected some tags in response: %v", tagsResponse.Tags)
	}

	if tagsResponse.Tags[0] != tag {
		t.Fatalf("tag not as expected: %q != %q", tagsResponse.Tags[0], tag)
	}

	// ---------------
	// Delete by digest
	resp, err = httpDelete(manifestDigestURL)
	checkErr(t, err, "deleting manifest by digest")

	checkResponse(t, "deleting manifest with tag", resp, http.StatusAccepted)
	checkHeaders(t, resp, http.Header{
		"Content-Length": []string{"0"},
	})

	// Ensure that the tag is not listed.
	resp, err = http.Get(tagsURL)
	if err != nil {
		t.Fatalf("unexpected error getting unknown tags: %v", err)
	}
	defer resp.Body.Close()

	dec = json.NewDecoder(resp.Body)
	if err := dec.Decode(&tagsResponse); err != nil {
		t.Fatalf("unexpected error decoding error response: %v", err)
	}

	if tagsResponse.Name != imageName.Name() {
		t.Fatalf("tags name should match image name: %v != %v", tagsResponse.Name, imageName)
	}

	if len(tagsResponse.Tags) != 0 {
		t.Fatalf("expected 0 tags in response: %v", tagsResponse.Tags)
	}

}

type testEnv struct {
	pk      libtrust.PrivateKey
	ctx     context.Context
	config  configuration.Configuration
	app     *App
	server  *httptest.Server
	builder *v2.URLBuilder
}

func newTestEnvMirror(t *testing.T, deleteEnabled bool) *testEnv {
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
			"delete":     configuration.Parameters{"enabled": deleteEnabled},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
		Proxy: configuration.Proxy{
			RemoteURL: "http://example.com",
		},
	}

	return newTestEnvWithConfig(t, &config)

}

func newTestEnv(t *testing.T, deleteEnabled bool) *testEnv {
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
			"delete":     configuration.Parameters{"enabled": deleteEnabled},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
	}

	config.HTTP.Headers = headerConfig

	return newTestEnvWithConfig(t, &config)
}

func newTestEnvWithConfig(t *testing.T, config *configuration.Configuration) *testEnv {
	ctx := context.Background()

	app := NewApp(ctx, config)
	server := httptest.NewServer(handlers.CombinedLoggingHandler(os.Stderr, app))
	builder, err := v2.NewURLBuilderFromString(server.URL+config.HTTP.Prefix, false)

	if err != nil {
		t.Fatalf("error creating url builder: %v", err)
	}

	pk, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("unexpected error generating private key: %v", err)
	}

	return &testEnv{
		pk:      pk,
		ctx:     ctx,
		config:  *config,
		app:     app,
		server:  server,
		builder: builder,
	}
}

func (t *testEnv) Shutdown() {
	t.server.CloseClientConnections()
	t.server.Close()
}

func putManifest(t *testing.T, msg, url, contentType string, v interface{}) *http.Response {
	var body []byte

	switch m := v.(type) {
	case *schema1.SignedManifest:
		_, pl, err := m.Payload()
		if err != nil {
			t.Fatalf("error getting payload: %v", err)
		}
		body = pl
	case *manifestlist.DeserializedManifestList:
		_, pl, err := m.Payload()
		if err != nil {
			t.Fatalf("error getting payload: %v", err)
		}
		body = pl
	default:
		var err error
		body, err = json.MarshalIndent(v, "", "   ")
		if err != nil {
			t.Fatalf("unexpected error marshaling %v: %v", v, err)
		}
	}

	req, err := http.NewRequest("PUT", url, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("error creating request for %s: %v", msg, err)
	}

	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("error doing put request while %s: %v", msg, err)
	}

	return resp
}

func startPushLayer(t *testing.T, env *testEnv, name reference.Named) (location string, uuid string) {
	layerUploadURL, err := env.builder.BuildBlobUploadURL(name)
	if err != nil {
		t.Fatalf("unexpected error building layer upload url: %v", err)
	}

	u, err := url.Parse(layerUploadURL)
	if err != nil {
		t.Fatalf("error parsing layer upload URL: %v", err)
	}

	base, err := url.Parse(env.server.URL)
	if err != nil {
		t.Fatalf("error parsing server URL: %v", err)
	}

	layerUploadURL = base.ResolveReference(u).String()
	resp, err := http.Post(layerUploadURL, "", nil)
	if err != nil {
		t.Fatalf("unexpected error starting layer push: %v", err)
	}

	defer resp.Body.Close()

	checkResponse(t, fmt.Sprintf("pushing starting layer push %v", name.String()), resp, http.StatusAccepted)

	u, err = url.Parse(resp.Header.Get("Location"))
	if err != nil {
		t.Fatalf("error parsing location header: %v", err)
	}

	uuid = path.Base(u.Path)
	checkHeaders(t, resp, http.Header{
		"Location":           []string{"*"},
		"Content-Length":     []string{"0"},
		"Docker-Upload-UUID": []string{uuid},
	})

	return resp.Header.Get("Location"), uuid
}

// doPushLayer pushes the layer content returning the url on success returning
// the response. If you're only expecting a successful response, use pushLayer.
func doPushLayer(t *testing.T, ub *v2.URLBuilder, name reference.Named, dgst digest.Digest, uploadURLBase string, body io.Reader) (*http.Response, error) {
	u, err := url.Parse(uploadURLBase)
	if err != nil {
		t.Fatalf("unexpected error parsing pushLayer url: %v", err)
	}

	u.RawQuery = url.Values{
		"_state": u.Query()["_state"],
		"digest": []string{dgst.String()},
	}.Encode()

	uploadURL := u.String()

	// Just do a monolithic upload
	req, err := http.NewRequest("PUT", uploadURL, body)
	if err != nil {
		t.Fatalf("unexpected error creating new request: %v", err)
	}

	return http.DefaultClient.Do(req)
}

// pushLayer pushes the layer content returning the url on success.
func pushLayer(t *testing.T, ub *v2.URLBuilder, name reference.Named, dgst digest.Digest, uploadURLBase string, body io.Reader) string {
	digester := digest.Canonical.Digester()

	resp, err := doPushLayer(t, ub, name, dgst, uploadURLBase, io.TeeReader(body, digester.Hash()))
	if err != nil {
		t.Fatalf("unexpected error doing push layer request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "putting monolithic chunk", resp, http.StatusCreated)

	if err != nil {
		t.Fatalf("error generating sha256 digest of body")
	}

	sha256Dgst := digester.Digest()

	ref, _ := reference.WithDigest(name, sha256Dgst)
	expectedLayerURL, err := ub.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("error building expected layer url: %v", err)
	}

	checkHeaders(t, resp, http.Header{
		"Location":              []string{expectedLayerURL},
		"Content-Length":        []string{"0"},
		"Docker-Content-Digest": []string{sha256Dgst.String()},
	})

	return resp.Header.Get("Location")
}

func finishUpload(t *testing.T, ub *v2.URLBuilder, name reference.Named, uploadURLBase string, dgst digest.Digest) string {
	resp, err := doPushLayer(t, ub, name, dgst, uploadURLBase, nil)
	if err != nil {
		t.Fatalf("unexpected error doing push layer request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "putting monolithic chunk", resp, http.StatusCreated)

	ref, _ := reference.WithDigest(name, dgst)
	expectedLayerURL, err := ub.BuildBlobURL(ref)
	if err != nil {
		t.Fatalf("error building expected layer url: %v", err)
	}

	checkHeaders(t, resp, http.Header{
		"Location":              []string{expectedLayerURL},
		"Content-Length":        []string{"0"},
		"Docker-Content-Digest": []string{dgst.String()},
	})

	return resp.Header.Get("Location")
}

func doPushChunk(t *testing.T, uploadURLBase string, body io.Reader) (*http.Response, digest.Digest, error) {
	u, err := url.Parse(uploadURLBase)
	if err != nil {
		t.Fatalf("unexpected error parsing pushLayer url: %v", err)
	}

	u.RawQuery = url.Values{
		"_state": u.Query()["_state"],
	}.Encode()

	uploadURL := u.String()

	digester := digest.Canonical.Digester()

	req, err := http.NewRequest("PATCH", uploadURL, io.TeeReader(body, digester.Hash()))
	if err != nil {
		t.Fatalf("unexpected error creating new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := http.DefaultClient.Do(req)

	return resp, digester.Digest(), err
}

func pushChunk(t *testing.T, ub *v2.URLBuilder, name reference.Named, uploadURLBase string, body io.Reader, length int64) (string, digest.Digest) {
	resp, dgst, err := doPushChunk(t, uploadURLBase, body)
	if err != nil {
		t.Fatalf("unexpected error doing push layer request: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, "putting chunk", resp, http.StatusAccepted)

	if err != nil {
		t.Fatalf("error generating sha256 digest of body")
	}

	checkHeaders(t, resp, http.Header{
		"Range":          []string{fmt.Sprintf("0-%d", length-1)},
		"Content-Length": []string{"0"},
	})

	return resp.Header.Get("Location"), dgst
}

func checkResponse(t *testing.T, msg string, resp *http.Response, expectedStatus int) {
	if resp.StatusCode != expectedStatus {
		t.Logf("unexpected status %s: %v != %v", msg, resp.StatusCode, expectedStatus)
		maybeDumpResponse(t, resp)

		t.FailNow()
	}

	// We expect the headers included in the configuration, unless the
	// status code is 405 (Method Not Allowed), which means the handler
	// doesn't even get called.
	if resp.StatusCode != 405 && !reflect.DeepEqual(resp.Header["X-Content-Type-Options"], []string{"nosniff"}) {
		t.Logf("missing or incorrect header X-Content-Type-Options %s", msg)
		maybeDumpResponse(t, resp)

		t.FailNow()
	}
}

// checkBodyHasErrorCodes ensures the body is an error body and has the
// expected error codes, returning the error structure, the json slice and a
// count of the errors by code.
func checkBodyHasErrorCodes(t *testing.T, msg string, resp *http.Response, errorCodes ...errcode.ErrorCode) (errcode.Errors, []byte, map[errcode.ErrorCode]int) {
	p, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("unexpected error reading body %s: %v", msg, err)
	}

	var errs errcode.Errors
	if err := json.Unmarshal(p, &errs); err != nil {
		t.Fatalf("unexpected error decoding error response: %v", err)
	}

	if len(errs) == 0 {
		t.Fatalf("expected errors in response")
	}

	// TODO(stevvooe): Shoot. The error setup is not working out. The content-
	// type headers are being set after writing the status code.
	// if resp.Header.Get("Content-Type") != "application/json; charset=utf-8" {
	// 	t.Fatalf("unexpected content type: %v != 'application/json'",
	// 		resp.Header.Get("Content-Type"))
	// }

	expected := map[errcode.ErrorCode]struct{}{}
	counts := map[errcode.ErrorCode]int{}

	// Initialize map with zeros for expected
	for _, code := range errorCodes {
		expected[code] = struct{}{}
		counts[code] = 0
	}

	for _, e := range errs {
		err, ok := e.(errcode.ErrorCoder)
		if !ok {
			t.Fatalf("not an ErrorCoder: %#v", e)
		}
		if _, ok := expected[err.ErrorCode()]; !ok {
			t.Fatalf("unexpected error code %v encountered during %s: %s ", err.ErrorCode(), msg, string(p))
		}
		counts[err.ErrorCode()]++
	}

	// Ensure that counts of expected errors were all non-zero
	for code := range expected {
		if counts[code] == 0 {
			t.Fatalf("expected error code %v not encounterd during %s: %s", code, msg, string(p))
		}
	}

	return errs, p, counts
}

func maybeDumpResponse(t *testing.T, resp *http.Response) {
	if d, err := httputil.DumpResponse(resp, true); err != nil {
		t.Logf("error dumping response: %v", err)
	} else {
		t.Logf("response:\n%s", string(d))
	}
}

// matchHeaders checks that the response has at least the headers. If not, the
// test will fail. If a passed in header value is "*", any non-zero value will
// suffice as a match.
func checkHeaders(t *testing.T, resp *http.Response, headers http.Header) {
	for k, vs := range headers {
		if resp.Header.Get(k) == "" {
			t.Fatalf("response missing header %q", k)
		}

		for _, v := range vs {
			if v == "*" {
				// Just ensure there is some value.
				if len(resp.Header[http.CanonicalHeaderKey(k)]) > 0 {
					continue
				}
			}

			for _, hv := range resp.Header[http.CanonicalHeaderKey(k)] {
				if hv != v {
					t.Fatalf("%+v %v header value not matched in response: %q != %q", resp.Header, k, hv, v)
				}
			}
		}
	}
}

func checkErr(t *testing.T, err error, msg string) {
	if err != nil {
		t.Fatalf("unexpected error %s: %v", msg, err)
	}
}

func createRepository(env *testEnv, t *testing.T, imageName string, tag string) digest.Digest {
	imageNameRef, err := reference.WithName(imageName)
	if err != nil {
		t.Fatalf("unable to parse reference: %v", err)
	}

	unsignedManifest := &schema1.Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name: imageName,
		Tag:  tag,
		FSLayers: []schema1.FSLayer{
			{
				BlobSum: "asdf",
			},
		},
		History: []schema1.History{
			{
				V1Compatibility: "",
			},
		},
	}

	// Push 2 random layers
	expectedLayers := make(map[digest.Digest]io.ReadSeeker)

	for i := range unsignedManifest.FSLayers {
		rs, dgstStr, err := testutil.CreateRandomTarFile()
		if err != nil {
			t.Fatalf("error creating random layer %d: %v", i, err)
		}
		dgst := digest.Digest(dgstStr)

		expectedLayers[dgst] = rs
		unsignedManifest.FSLayers[i].BlobSum = dgst
		uploadURLBase, _ := startPushLayer(t, env, imageNameRef)
		pushLayer(t, env.builder, imageNameRef, dgst, uploadURLBase, rs)
	}

	signedManifest, err := schema1.Sign(unsignedManifest, env.pk)
	if err != nil {
		t.Fatalf("unexpected error signing manifest: %v", err)
	}

	dgst := digest.FromBytes(signedManifest.Canonical)

	// Create this repository by tag to ensure the tag mapping is made in the registry
	tagRef, _ := reference.WithTag(imageNameRef, tag)
	manifestDigestURL, err := env.builder.BuildManifestURL(tagRef)
	checkErr(t, err, "building manifest url")

	digestRef, _ := reference.WithDigest(imageNameRef, dgst)
	location, err := env.builder.BuildManifestURL(digestRef)
	checkErr(t, err, "building location URL")

	resp := putManifest(t, "putting signed manifest", manifestDigestURL, "", signedManifest)
	checkResponse(t, "putting signed manifest", resp, http.StatusCreated)
	checkHeaders(t, resp, http.Header{
		"Location":              []string{location},
		"Docker-Content-Digest": []string{dgst.String()},
	})
	return dgst
}

// Test mutation operations on a registry configured as a cache.  Ensure that they return
// appropriate errors.
func TestRegistryAsCacheMutationAPIs(t *testing.T) {
	deleteEnabled := true
	env := newTestEnvMirror(t, deleteEnabled)
	defer env.Shutdown()

	imageName, _ := reference.WithName("foo/bar")
	tag := "latest"
	tagRef, _ := reference.WithTag(imageName, tag)
	manifestURL, err := env.builder.BuildManifestURL(tagRef)
	if err != nil {
		t.Fatalf("unexpected error building base url: %v", err)
	}

	// Manifest upload
	m := &schema1.Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name:     imageName.Name(),
		Tag:      tag,
		FSLayers: []schema1.FSLayer{},
		History:  []schema1.History{},
	}

	sm, err := schema1.Sign(m, env.pk)
	if err != nil {
		t.Fatalf("error signing manifest: %v", err)
	}

	resp := putManifest(t, "putting unsigned manifest", manifestURL, "", sm)
	checkResponse(t, "putting signed manifest to cache", resp, errcode.ErrorCodeUnsupported.Descriptor().HTTPStatusCode)

	// Manifest Delete
	resp, err = httpDelete(manifestURL)
	checkResponse(t, "deleting signed manifest from cache", resp, errcode.ErrorCodeUnsupported.Descriptor().HTTPStatusCode)

	// Blob upload initialization
	layerUploadURL, err := env.builder.BuildBlobUploadURL(imageName)
	if err != nil {
		t.Fatalf("unexpected error building layer upload url: %v", err)
	}

	resp, err = http.Post(layerUploadURL, "", nil)
	if err != nil {
		t.Fatalf("unexpected error starting layer push: %v", err)
	}
	defer resp.Body.Close()

	checkResponse(t, fmt.Sprintf("starting layer push to cache %v", imageName), resp, errcode.ErrorCodeUnsupported.Descriptor().HTTPStatusCode)

	// Blob Delete
	ref, _ := reference.WithDigest(imageName, digestSha256EmptyTar)
	blobURL, err := env.builder.BuildBlobURL(ref)
	resp, err = httpDelete(blobURL)
	checkResponse(t, "deleting blob from cache", resp, errcode.ErrorCodeUnsupported.Descriptor().HTTPStatusCode)

}

// TestCheckContextNotifier makes sure the API endpoints get a ResponseWriter
// that implements http.ContextNotifier.
func TestCheckContextNotifier(t *testing.T) {
	env := newTestEnv(t, false)
	defer env.Shutdown()

	// Register a new endpoint for testing
	env.app.router.Handle("/unittest/{name}/", env.app.dispatcher(func(ctx *Context, r *http.Request) http.Handler {
		return handlers.MethodHandler{
			"GET": http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, ok := w.(http.CloseNotifier); !ok {
					t.Fatal("could not cast ResponseWriter to CloseNotifier")
				}
				w.WriteHeader(200)
			}),
		}
	}))

	resp, err := http.Get(env.server.URL + "/unittest/reponame/")
	if err != nil {
		t.Fatalf("unexpected error issuing request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("wrong status code - expected 200, got %d", resp.StatusCode)
	}
}

func TestProxyManifestGetByTag(t *testing.T) {
	truthConfig := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
			"maintenance": configuration.Parameters{"uploadpurging": map[interface{}]interface{}{
				"enabled": false,
			}},
		},
	}
	truthConfig.HTTP.Headers = headerConfig

	imageName, _ := reference.WithName("foo/bar")
	tag := "latest"

	truthEnv := newTestEnvWithConfig(t, &truthConfig)
	defer truthEnv.Shutdown()
	// create a repository in the truth registry
	dgst := createRepository(truthEnv, t, imageName.Name(), tag)

	proxyConfig := configuration.Configuration{
		Storage: configuration.Storage{
			"testdriver": configuration.Parameters{},
		},
		Proxy: configuration.Proxy{
			RemoteURL: truthEnv.server.URL,
		},
	}
	proxyConfig.HTTP.Headers = headerConfig

	proxyEnv := newTestEnvWithConfig(t, &proxyConfig)
	defer proxyEnv.Shutdown()

	digestRef, _ := reference.WithDigest(imageName, dgst)
	manifestDigestURL, err := proxyEnv.builder.BuildManifestURL(digestRef)
	checkErr(t, err, "building manifest url")

	resp, err := http.Get(manifestDigestURL)
	checkErr(t, err, "fetching manifest from proxy by digest")
	defer resp.Body.Close()

	tagRef, _ := reference.WithTag(imageName, tag)
	manifestTagURL, err := proxyEnv.builder.BuildManifestURL(tagRef)
	checkErr(t, err, "building manifest url")

	resp, err = http.Get(manifestTagURL)
	checkErr(t, err, "fetching manifest from proxy by tag")
	defer resp.Body.Close()
	checkResponse(t, "fetching manifest from proxy by tag", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{dgst.String()},
	})

	// Create another manifest in the remote with the same image/tag pair
	newDigest := createRepository(truthEnv, t, imageName.Name(), tag)
	if dgst == newDigest {
		t.Fatalf("non-random test data")
	}

	// fetch it with the same proxy URL as before.  Ensure the updated content is at the same tag
	resp, err = http.Get(manifestTagURL)
	checkErr(t, err, "fetching manifest from proxy by tag")
	defer resp.Body.Close()
	checkResponse(t, "fetching manifest from proxy by tag", resp, http.StatusOK)
	checkHeaders(t, resp, http.Header{
		"Docker-Content-Digest": []string{newDigest.String()},
	})
}
