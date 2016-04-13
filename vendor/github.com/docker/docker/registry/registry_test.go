package registry

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"testing"

	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/cliconfig"
)

var (
	token = []string{"fake-token"}
)

const (
	imageID = "42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d"
	REPO    = "foo42/bar"
)

func spawnTestRegistrySession(t *testing.T) *Session {
	authConfig := &cliconfig.AuthConfig{}
	endpoint, err := NewEndpoint(makeIndex("/v1/"), nil)
	if err != nil {
		t.Fatal(err)
	}
	var tr http.RoundTripper = debugTransport{NewTransport(nil), t.Log}
	tr = transport.NewTransport(AuthTransport(tr, authConfig, false), DockerHeaders(nil)...)
	client := HTTPClient(tr)
	r, err := NewSession(client, authConfig, endpoint)
	if err != nil {
		t.Fatal(err)
	}
	// In a normal scenario for the v1 registry, the client should send a `X-Docker-Token: true`
	// header while authenticating, in order to retrieve a token that can be later used to
	// perform authenticated actions.
	//
	// The mock v1 registry does not support that, (TODO(tiborvass): support it), instead,
	// it will consider authenticated any request with the header `X-Docker-Token: fake-token`.
	//
	// Because we know that the client's transport is an `*authTransport` we simply cast it,
	// in order to set the internal cached token to the fake token, and thus send that fake token
	// upon every subsequent requests.
	r.client.Transport.(*authTransport).token = token
	return r
}

func TestPingRegistryEndpoint(t *testing.T) {
	testPing := func(index *IndexInfo, expectedStandalone bool, assertMessage string) {
		ep, err := NewEndpoint(index, nil)
		if err != nil {
			t.Fatal(err)
		}
		regInfo, err := ep.Ping()
		if err != nil {
			t.Fatal(err)
		}

		assertEqual(t, regInfo.Standalone, expectedStandalone, assertMessage)
	}

	testPing(makeIndex("/v1/"), true, "Expected standalone to be true (default)")
	testPing(makeHttpsIndex("/v1/"), true, "Expected standalone to be true (default)")
	testPing(makePublicIndex(), false, "Expected standalone to be false for public index")
}

func TestEndpoint(t *testing.T) {
	// Simple wrapper to fail test if err != nil
	expandEndpoint := func(index *IndexInfo) *Endpoint {
		endpoint, err := NewEndpoint(index, nil)
		if err != nil {
			t.Fatal(err)
		}
		return endpoint
	}

	assertInsecureIndex := func(index *IndexInfo) {
		index.Secure = true
		_, err := NewEndpoint(index, nil)
		assertNotEqual(t, err, nil, index.Name+": Expected error for insecure index")
		assertEqual(t, strings.Contains(err.Error(), "insecure-registry"), true, index.Name+": Expected insecure-registry  error for insecure index")
		index.Secure = false
	}

	assertSecureIndex := func(index *IndexInfo) {
		index.Secure = true
		_, err := NewEndpoint(index, nil)
		assertNotEqual(t, err, nil, index.Name+": Expected cert error for secure index")
		assertEqual(t, strings.Contains(err.Error(), "certificate signed by unknown authority"), true, index.Name+": Expected cert error for secure index")
		index.Secure = false
	}

	index := &IndexInfo{}
	index.Name = makeURL("/v1/")
	endpoint := expandEndpoint(index)
	assertEqual(t, endpoint.String(), index.Name, "Expected endpoint to be "+index.Name)
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertInsecureIndex(index)

	index.Name = makeURL("")
	endpoint = expandEndpoint(index)
	assertEqual(t, endpoint.String(), index.Name+"/v1/", index.Name+": Expected endpoint to be "+index.Name+"/v1/")
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertInsecureIndex(index)

	httpURL := makeURL("")
	index.Name = strings.SplitN(httpURL, "://", 2)[1]
	endpoint = expandEndpoint(index)
	assertEqual(t, endpoint.String(), httpURL+"/v1/", index.Name+": Expected endpoint to be "+httpURL+"/v1/")
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertInsecureIndex(index)

	index.Name = makeHttpsURL("/v1/")
	endpoint = expandEndpoint(index)
	assertEqual(t, endpoint.String(), index.Name, "Expected endpoint to be "+index.Name)
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertSecureIndex(index)

	index.Name = makeHttpsURL("")
	endpoint = expandEndpoint(index)
	assertEqual(t, endpoint.String(), index.Name+"/v1/", index.Name+": Expected endpoint to be "+index.Name+"/v1/")
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertSecureIndex(index)

	httpsURL := makeHttpsURL("")
	index.Name = strings.SplitN(httpsURL, "://", 2)[1]
	endpoint = expandEndpoint(index)
	assertEqual(t, endpoint.String(), httpsURL+"/v1/", index.Name+": Expected endpoint to be "+httpsURL+"/v1/")
	if endpoint.Version != APIVersion1 {
		t.Fatal("Expected endpoint to be v1")
	}
	assertSecureIndex(index)

	badEndpoints := []string{
		"http://127.0.0.1/v1/",
		"https://127.0.0.1/v1/",
		"http://127.0.0.1",
		"https://127.0.0.1",
		"127.0.0.1",
	}
	for _, address := range badEndpoints {
		index.Name = address
		_, err := NewEndpoint(index, nil)
		checkNotEqual(t, err, nil, "Expected error while expanding bad endpoint")
	}
}

func TestGetRemoteHistory(t *testing.T) {
	r := spawnTestRegistrySession(t)
	hist, err := r.GetRemoteHistory(imageID, makeURL("/v1/"))
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(hist), 2, "Expected 2 images in history")
	assertEqual(t, hist[0], imageID, "Expected "+imageID+"as first ancestry")
	assertEqual(t, hist[1], "77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20",
		"Unexpected second ancestry")
}

func TestLookupRemoteImage(t *testing.T) {
	r := spawnTestRegistrySession(t)
	err := r.LookupRemoteImage(imageID, makeURL("/v1/"))
	assertEqual(t, err, nil, "Expected error of remote lookup to nil")
	if err := r.LookupRemoteImage("abcdef", makeURL("/v1/")); err == nil {
		t.Fatal("Expected error of remote lookup to not nil")
	}
}

func TestGetRemoteImageJSON(t *testing.T) {
	r := spawnTestRegistrySession(t)
	json, size, err := r.GetRemoteImageJSON(imageID, makeURL("/v1/"))
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, size, 154, "Expected size 154")
	if len(json) <= 0 {
		t.Fatal("Expected non-empty json")
	}

	_, _, err = r.GetRemoteImageJSON("abcdef", makeURL("/v1/"))
	if err == nil {
		t.Fatal("Expected image not found error")
	}
}

func TestGetRemoteImageLayer(t *testing.T) {
	r := spawnTestRegistrySession(t)
	data, err := r.GetRemoteImageLayer(imageID, makeURL("/v1/"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if data == nil {
		t.Fatal("Expected non-nil data result")
	}

	_, err = r.GetRemoteImageLayer("abcdef", makeURL("/v1/"), 0)
	if err == nil {
		t.Fatal("Expected image not found error")
	}
}

func TestGetRemoteTag(t *testing.T) {
	r := spawnTestRegistrySession(t)
	tag, err := r.GetRemoteTag([]string{makeURL("/v1/")}, REPO, "test")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, tag, imageID, "Expected tag test to map to "+imageID)

	_, err = r.GetRemoteTag([]string{makeURL("/v1/")}, "foo42/baz", "foo")
	if err != ErrRepoNotFound {
		t.Fatal("Expected ErrRepoNotFound error when fetching tag for bogus repo")
	}
}

func TestGetRemoteTags(t *testing.T) {
	r := spawnTestRegistrySession(t)
	tags, err := r.GetRemoteTags([]string{makeURL("/v1/")}, REPO)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(tags), 2, "Expected two tags")
	assertEqual(t, tags["latest"], imageID, "Expected tag latest to map to "+imageID)
	assertEqual(t, tags["test"], imageID, "Expected tag test to map to "+imageID)

	_, err = r.GetRemoteTags([]string{makeURL("/v1/")}, "foo42/baz")
	if err != ErrRepoNotFound {
		t.Fatal("Expected ErrRepoNotFound error when fetching tags for bogus repo")
	}
}

func TestGetRepositoryData(t *testing.T) {
	r := spawnTestRegistrySession(t)
	parsedURL, err := url.Parse(makeURL("/v1/"))
	if err != nil {
		t.Fatal(err)
	}
	host := "http://" + parsedURL.Host + "/v1/"
	data, err := r.GetRepositoryData("foo42/bar")
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, len(data.ImgList), 2, "Expected 2 images in ImgList")
	assertEqual(t, len(data.Endpoints), 2,
		fmt.Sprintf("Expected 2 endpoints in Endpoints, found %d instead", len(data.Endpoints)))
	assertEqual(t, data.Endpoints[0], host,
		fmt.Sprintf("Expected first endpoint to be %s but found %s instead", host, data.Endpoints[0]))
	assertEqual(t, data.Endpoints[1], "http://test.example.com/v1/",
		fmt.Sprintf("Expected first endpoint to be http://test.example.com/v1/ but found %s instead", data.Endpoints[1]))

}

func TestPushImageJSONRegistry(t *testing.T) {
	r := spawnTestRegistrySession(t)
	imgData := &ImgData{
		ID:       "77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20",
		Checksum: "sha256:1ac330d56e05eef6d438586545ceff7550d3bdcb6b19961f12c5ba714ee1bb37",
	}

	err := r.PushImageJSONRegistry(imgData, []byte{0x42, 0xdf, 0x0}, makeURL("/v1/"))
	if err != nil {
		t.Fatal(err)
	}
}

func TestPushImageLayerRegistry(t *testing.T) {
	r := spawnTestRegistrySession(t)
	layer := strings.NewReader("")
	_, _, err := r.PushImageLayerRegistry(imageID, layer, makeURL("/v1/"), []byte{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestValidateRepositoryName(t *testing.T) {
	validRepoNames := []string{
		"docker/docker",
		"library/debian",
		"debian",
		"docker.io/docker/docker",
		"docker.io/library/debian",
		"docker.io/debian",
		"index.docker.io/docker/docker",
		"index.docker.io/library/debian",
		"index.docker.io/debian",
		"127.0.0.1:5000/docker/docker",
		"127.0.0.1:5000/library/debian",
		"127.0.0.1:5000/debian",
		"thisisthesongthatneverendsitgoesonandonandonthisisthesongthatnev",
	}
	invalidRepoNames := []string{
		"https://github.com/docker/docker",
		"docker/Docker",
		"-docker",
		"-docker/docker",
		"-docker.io/docker/docker",
		"docker///docker",
		"docker.io/docker/Docker",
		"docker.io/docker///docker",
		"1a3f5e7d9c1b3a5f7e9d1c3b5a7f9e1d3c5b7a9f1e3d5d7c9b1a3f5e7d9c1b3a",
		"docker.io/1a3f5e7d9c1b3a5f7e9d1c3b5a7f9e1d3c5b7a9f1e3d5d7c9b1a3f5e7d9c1b3a",
	}

	for _, name := range invalidRepoNames {
		err := ValidateRepositoryName(name)
		assertNotEqual(t, err, nil, "Expected invalid repo name: "+name)
	}

	for _, name := range validRepoNames {
		err := ValidateRepositoryName(name)
		assertEqual(t, err, nil, "Expected valid repo name: "+name)
	}

	err := ValidateRepositoryName(invalidRepoNames[0])
	assertEqual(t, err, ErrInvalidRepositoryName, "Expected ErrInvalidRepositoryName: "+invalidRepoNames[0])
}

func TestParseRepositoryInfo(t *testing.T) {
	expectedRepoInfos := map[string]RepositoryInfo{
		"fooo/bar": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "fooo/bar",
			LocalName:     "fooo/bar",
			CanonicalName: "docker.io/fooo/bar",
			Official:      false,
		},
		"library/ubuntu": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "library/ubuntu",
			LocalName:     "ubuntu",
			CanonicalName: "docker.io/library/ubuntu",
			Official:      true,
		},
		"nonlibrary/ubuntu": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "nonlibrary/ubuntu",
			LocalName:     "nonlibrary/ubuntu",
			CanonicalName: "docker.io/nonlibrary/ubuntu",
			Official:      false,
		},
		"ubuntu": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "library/ubuntu",
			LocalName:     "ubuntu",
			CanonicalName: "docker.io/library/ubuntu",
			Official:      true,
		},
		"other/library": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "other/library",
			LocalName:     "other/library",
			CanonicalName: "docker.io/other/library",
			Official:      false,
		},
		"127.0.0.1:8000/private/moonbase": {
			Index: &IndexInfo{
				Name:     "127.0.0.1:8000",
				Official: false,
			},
			RemoteName:    "private/moonbase",
			LocalName:     "127.0.0.1:8000/private/moonbase",
			CanonicalName: "127.0.0.1:8000/private/moonbase",
			Official:      false,
		},
		"127.0.0.1:8000/privatebase": {
			Index: &IndexInfo{
				Name:     "127.0.0.1:8000",
				Official: false,
			},
			RemoteName:    "privatebase",
			LocalName:     "127.0.0.1:8000/privatebase",
			CanonicalName: "127.0.0.1:8000/privatebase",
			Official:      false,
		},
		"localhost:8000/private/moonbase": {
			Index: &IndexInfo{
				Name:     "localhost:8000",
				Official: false,
			},
			RemoteName:    "private/moonbase",
			LocalName:     "localhost:8000/private/moonbase",
			CanonicalName: "localhost:8000/private/moonbase",
			Official:      false,
		},
		"localhost:8000/privatebase": {
			Index: &IndexInfo{
				Name:     "localhost:8000",
				Official: false,
			},
			RemoteName:    "privatebase",
			LocalName:     "localhost:8000/privatebase",
			CanonicalName: "localhost:8000/privatebase",
			Official:      false,
		},
		"example.com/private/moonbase": {
			Index: &IndexInfo{
				Name:     "example.com",
				Official: false,
			},
			RemoteName:    "private/moonbase",
			LocalName:     "example.com/private/moonbase",
			CanonicalName: "example.com/private/moonbase",
			Official:      false,
		},
		"example.com/privatebase": {
			Index: &IndexInfo{
				Name:     "example.com",
				Official: false,
			},
			RemoteName:    "privatebase",
			LocalName:     "example.com/privatebase",
			CanonicalName: "example.com/privatebase",
			Official:      false,
		},
		"example.com:8000/private/moonbase": {
			Index: &IndexInfo{
				Name:     "example.com:8000",
				Official: false,
			},
			RemoteName:    "private/moonbase",
			LocalName:     "example.com:8000/private/moonbase",
			CanonicalName: "example.com:8000/private/moonbase",
			Official:      false,
		},
		"example.com:8000/privatebase": {
			Index: &IndexInfo{
				Name:     "example.com:8000",
				Official: false,
			},
			RemoteName:    "privatebase",
			LocalName:     "example.com:8000/privatebase",
			CanonicalName: "example.com:8000/privatebase",
			Official:      false,
		},
		"localhost/private/moonbase": {
			Index: &IndexInfo{
				Name:     "localhost",
				Official: false,
			},
			RemoteName:    "private/moonbase",
			LocalName:     "localhost/private/moonbase",
			CanonicalName: "localhost/private/moonbase",
			Official:      false,
		},
		"localhost/privatebase": {
			Index: &IndexInfo{
				Name:     "localhost",
				Official: false,
			},
			RemoteName:    "privatebase",
			LocalName:     "localhost/privatebase",
			CanonicalName: "localhost/privatebase",
			Official:      false,
		},
		INDEXNAME + "/public/moonbase": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "public/moonbase",
			LocalName:     "public/moonbase",
			CanonicalName: "docker.io/public/moonbase",
			Official:      false,
		},
		"index." + INDEXNAME + "/public/moonbase": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "public/moonbase",
			LocalName:     "public/moonbase",
			CanonicalName: "docker.io/public/moonbase",
			Official:      false,
		},
		"ubuntu-12.04-base": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "library/ubuntu-12.04-base",
			LocalName:     "ubuntu-12.04-base",
			CanonicalName: "docker.io/library/ubuntu-12.04-base",
			Official:      true,
		},
		INDEXNAME + "/ubuntu-12.04-base": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "library/ubuntu-12.04-base",
			LocalName:     "ubuntu-12.04-base",
			CanonicalName: "docker.io/library/ubuntu-12.04-base",
			Official:      true,
		},
		"index." + INDEXNAME + "/ubuntu-12.04-base": {
			Index: &IndexInfo{
				Name:     INDEXNAME,
				Official: true,
			},
			RemoteName:    "library/ubuntu-12.04-base",
			LocalName:     "ubuntu-12.04-base",
			CanonicalName: "docker.io/library/ubuntu-12.04-base",
			Official:      true,
		},
	}

	for reposName, expectedRepoInfo := range expectedRepoInfos {
		repoInfo, err := ParseRepositoryInfo(reposName)
		if err != nil {
			t.Error(err)
		} else {
			checkEqual(t, repoInfo.Index.Name, expectedRepoInfo.Index.Name, reposName)
			checkEqual(t, repoInfo.RemoteName, expectedRepoInfo.RemoteName, reposName)
			checkEqual(t, repoInfo.LocalName, expectedRepoInfo.LocalName, reposName)
			checkEqual(t, repoInfo.CanonicalName, expectedRepoInfo.CanonicalName, reposName)
			checkEqual(t, repoInfo.Index.Official, expectedRepoInfo.Index.Official, reposName)
			checkEqual(t, repoInfo.Official, expectedRepoInfo.Official, reposName)
		}
	}
}

func TestNewIndexInfo(t *testing.T) {
	testIndexInfo := func(config *ServiceConfig, expectedIndexInfos map[string]*IndexInfo) {
		for indexName, expectedIndexInfo := range expectedIndexInfos {
			index, err := config.NewIndexInfo(indexName)
			if err != nil {
				t.Fatal(err)
			} else {
				checkEqual(t, index.Name, expectedIndexInfo.Name, indexName+" name")
				checkEqual(t, index.Official, expectedIndexInfo.Official, indexName+" is official")
				checkEqual(t, index.Secure, expectedIndexInfo.Secure, indexName+" is secure")
				checkEqual(t, len(index.Mirrors), len(expectedIndexInfo.Mirrors), indexName+" mirrors")
			}
		}
	}

	config := NewServiceConfig(nil)
	noMirrors := make([]string, 0)
	expectedIndexInfos := map[string]*IndexInfo{
		INDEXNAME: {
			Name:     INDEXNAME,
			Official: true,
			Secure:   true,
			Mirrors:  noMirrors,
		},
		"index." + INDEXNAME: {
			Name:     INDEXNAME,
			Official: true,
			Secure:   true,
			Mirrors:  noMirrors,
		},
		"example.com": {
			Name:     "example.com",
			Official: false,
			Secure:   true,
			Mirrors:  noMirrors,
		},
		"127.0.0.1:5000": {
			Name:     "127.0.0.1:5000",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
	}
	testIndexInfo(config, expectedIndexInfos)

	publicMirrors := []string{"http://mirror1.local", "http://mirror2.local"}
	config = makeServiceConfig(publicMirrors, []string{"example.com"})

	expectedIndexInfos = map[string]*IndexInfo{
		INDEXNAME: {
			Name:     INDEXNAME,
			Official: true,
			Secure:   true,
			Mirrors:  publicMirrors,
		},
		"index." + INDEXNAME: {
			Name:     INDEXNAME,
			Official: true,
			Secure:   true,
			Mirrors:  publicMirrors,
		},
		"example.com": {
			Name:     "example.com",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"example.com:5000": {
			Name:     "example.com:5000",
			Official: false,
			Secure:   true,
			Mirrors:  noMirrors,
		},
		"127.0.0.1": {
			Name:     "127.0.0.1",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"127.0.0.1:5000": {
			Name:     "127.0.0.1:5000",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"other.com": {
			Name:     "other.com",
			Official: false,
			Secure:   true,
			Mirrors:  noMirrors,
		},
	}
	testIndexInfo(config, expectedIndexInfos)

	config = makeServiceConfig(nil, []string{"42.42.0.0/16"})
	expectedIndexInfos = map[string]*IndexInfo{
		"example.com": {
			Name:     "example.com",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"example.com:5000": {
			Name:     "example.com:5000",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"127.0.0.1": {
			Name:     "127.0.0.1",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"127.0.0.1:5000": {
			Name:     "127.0.0.1:5000",
			Official: false,
			Secure:   false,
			Mirrors:  noMirrors,
		},
		"other.com": {
			Name:     "other.com",
			Official: false,
			Secure:   true,
			Mirrors:  noMirrors,
		},
	}
	testIndexInfo(config, expectedIndexInfos)
}

func TestPushRegistryTag(t *testing.T) {
	r := spawnTestRegistrySession(t)
	err := r.PushRegistryTag("foo42/bar", imageID, "stable", makeURL("/v1/"))
	if err != nil {
		t.Fatal(err)
	}
}

func TestPushImageJSONIndex(t *testing.T) {
	r := spawnTestRegistrySession(t)
	imgData := []*ImgData{
		{
			ID:       "77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20",
			Checksum: "sha256:1ac330d56e05eef6d438586545ceff7550d3bdcb6b19961f12c5ba714ee1bb37",
		},
		{
			ID:       "42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d",
			Checksum: "sha256:bea7bf2e4bacd479344b737328db47b18880d09096e6674165533aa994f5e9f2",
		},
	}
	repoData, err := r.PushImageJSONIndex("foo42/bar", imgData, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if repoData == nil {
		t.Fatal("Expected RepositoryData object")
	}
	repoData, err = r.PushImageJSONIndex("foo42/bar", imgData, true, []string{r.indexEndpoint.String()})
	if err != nil {
		t.Fatal(err)
	}
	if repoData == nil {
		t.Fatal("Expected RepositoryData object")
	}
}

func TestSearchRepositories(t *testing.T) {
	r := spawnTestRegistrySession(t)
	results, err := r.SearchRepositories("fakequery")
	if err != nil {
		t.Fatal(err)
	}
	if results == nil {
		t.Fatal("Expected non-nil SearchResults object")
	}
	assertEqual(t, results.NumResults, 1, "Expected 1 search results")
	assertEqual(t, results.Query, "fakequery", "Expected 'fakequery' as query")
	assertEqual(t, results.Results[0].StarCount, 42, "Expected 'fakeimage' to have 42 stars")
}

func TestValidRemoteName(t *testing.T) {
	validRepositoryNames := []string{
		// Sanity check.
		"docker/docker",

		// Allow 64-character non-hexadecimal names (hexadecimal names are forbidden).
		"thisisthesongthatneverendsitgoesonandonandonthisisthesongthatnev",

		// Allow embedded hyphens.
		"docker-rules/docker",

		//Username doc and image name docker being tested.
		"doc/docker",

		// single character names are now allowed.
		"d/docker",
		"jess/t",
	}
	for _, repositoryName := range validRepositoryNames {
		if err := validateRemoteName(repositoryName); err != nil {
			t.Errorf("Repository name should be valid: %v. Error: %v", repositoryName, err)
		}
	}

	invalidRepositoryNames := []string{
		// Disallow capital letters.
		"docker/Docker",

		// Only allow one slash.
		"docker///docker",

		// Disallow 64-character hexadecimal.
		"1a3f5e7d9c1b3a5f7e9d1c3b5a7f9e1d3c5b7a9f1e3d5d7c9b1a3f5e7d9c1b3a",

		// Disallow leading and trailing hyphens in namespace.
		"-docker/docker",
		"docker-/docker",
		"-docker-/docker",

		// Don't allow underscores everywhere (as opposed to hyphens).
		"____/____",

		"_docker/_docker",

		// Disallow consecutive hyphens.
		"dock--er/docker",

		// No repository.
		"docker/",

		//namespace too long
		"this_is_not_a_valid_namespace_because_its_lenth_is_greater_than_255_this_is_not_a_valid_namespace_because_its_lenth_is_greater_than_255_this_is_not_a_valid_namespace_because_its_lenth_is_greater_than_255_this_is_not_a_valid_namespace_because_its_lenth_is_greater_than_255/docker",
	}
	for _, repositoryName := range invalidRepositoryNames {
		if err := validateRemoteName(repositoryName); err == nil {
			t.Errorf("Repository name should be invalid: %v", repositoryName)
		}
	}
}

func TestTrustedLocation(t *testing.T) {
	for _, url := range []string{"http://example.com", "https://example.com:7777", "http://docker.io", "http://test.docker.com", "https://fakedocker.com"} {
		req, _ := http.NewRequest("GET", url, nil)
		if trustedLocation(req) == true {
			t.Fatalf("'%s' shouldn't be detected as a trusted location", url)
		}
	}

	for _, url := range []string{"https://docker.io", "https://test.docker.com:80"} {
		req, _ := http.NewRequest("GET", url, nil)
		if trustedLocation(req) == false {
			t.Fatalf("'%s' should be detected as a trusted location", url)
		}
	}
}

func TestAddRequiredHeadersToRedirectedRequests(t *testing.T) {
	for _, urls := range [][]string{
		{"http://docker.io", "https://docker.com"},
		{"https://foo.docker.io:7777", "http://bar.docker.com"},
		{"https://foo.docker.io", "https://example.com"},
	} {
		reqFrom, _ := http.NewRequest("GET", urls[0], nil)
		reqFrom.Header.Add("Content-Type", "application/json")
		reqFrom.Header.Add("Authorization", "super_secret")
		reqTo, _ := http.NewRequest("GET", urls[1], nil)

		AddRequiredHeadersToRedirectedRequests(reqTo, []*http.Request{reqFrom})

		if len(reqTo.Header) != 1 {
			t.Fatalf("Expected 1 headers, got %d", len(reqTo.Header))
		}

		if reqTo.Header.Get("Content-Type") != "application/json" {
			t.Fatal("'Content-Type' should be 'application/json'")
		}

		if reqTo.Header.Get("Authorization") != "" {
			t.Fatal("'Authorization' should be empty")
		}
	}

	for _, urls := range [][]string{
		{"https://docker.io", "https://docker.com"},
		{"https://foo.docker.io:7777", "https://bar.docker.com"},
	} {
		reqFrom, _ := http.NewRequest("GET", urls[0], nil)
		reqFrom.Header.Add("Content-Type", "application/json")
		reqFrom.Header.Add("Authorization", "super_secret")
		reqTo, _ := http.NewRequest("GET", urls[1], nil)

		AddRequiredHeadersToRedirectedRequests(reqTo, []*http.Request{reqFrom})

		if len(reqTo.Header) != 2 {
			t.Fatalf("Expected 2 headers, got %d", len(reqTo.Header))
		}

		if reqTo.Header.Get("Content-Type") != "application/json" {
			t.Fatal("'Content-Type' should be 'application/json'")
		}

		if reqTo.Header.Get("Authorization") != "super_secret" {
			t.Fatal("'Authorization' should be 'super_secret'")
		}
	}
}

func TestIsSecureIndex(t *testing.T) {
	tests := []struct {
		addr               string
		insecureRegistries []string
		expected           bool
	}{
		{INDEXNAME, nil, true},
		{"example.com", []string{}, true},
		{"example.com", []string{"example.com"}, false},
		{"localhost", []string{"localhost:5000"}, false},
		{"localhost:5000", []string{"localhost:5000"}, false},
		{"localhost", []string{"example.com"}, false},
		{"127.0.0.1:5000", []string{"127.0.0.1:5000"}, false},
		{"localhost", nil, false},
		{"localhost:5000", nil, false},
		{"127.0.0.1", nil, false},
		{"localhost", []string{"example.com"}, false},
		{"127.0.0.1", []string{"example.com"}, false},
		{"example.com", nil, true},
		{"example.com", []string{"example.com"}, false},
		{"127.0.0.1", []string{"example.com"}, false},
		{"127.0.0.1:5000", []string{"example.com"}, false},
		{"example.com:5000", []string{"42.42.0.0/16"}, false},
		{"example.com", []string{"42.42.0.0/16"}, false},
		{"example.com:5000", []string{"42.42.42.42/8"}, false},
		{"127.0.0.1:5000", []string{"127.0.0.0/8"}, false},
		{"42.42.42.42:5000", []string{"42.1.1.1/8"}, false},
		{"invalid.domain.com", []string{"42.42.0.0/16"}, true},
		{"invalid.domain.com", []string{"invalid.domain.com"}, false},
		{"invalid.domain.com:5000", []string{"invalid.domain.com"}, true},
		{"invalid.domain.com:5000", []string{"invalid.domain.com:5000"}, false},
	}
	for _, tt := range tests {
		config := makeServiceConfig(nil, tt.insecureRegistries)
		if sec := config.isSecureIndex(tt.addr); sec != tt.expected {
			t.Errorf("isSecureIndex failed for %q %v, expected %v got %v", tt.addr, tt.insecureRegistries, tt.expected, sec)
		}
	}
}

type debugTransport struct {
	http.RoundTripper
	log func(...interface{})
}

func (tr debugTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	dump, err := httputil.DumpRequestOut(req, false)
	if err != nil {
		tr.log("could not dump request")
	}
	tr.log(string(dump))
	resp, err := tr.RoundTripper.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	dump, err = httputil.DumpResponse(resp, false)
	if err != nil {
		tr.log("could not dump response")
	}
	tr.log(string(dump))
	return resp, err
}
