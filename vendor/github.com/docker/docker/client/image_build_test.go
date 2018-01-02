package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/go-units"
)

func TestImageBuildError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImageBuild(context.Background(), nil, types.ImageBuildOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageBuild(t *testing.T) {
	v1 := "value1"
	v2 := "value2"
	emptyRegistryConfig := "bnVsbA=="
	buildCases := []struct {
		buildOptions           types.ImageBuildOptions
		expectedQueryParams    map[string]string
		expectedTags           []string
		expectedRegistryConfig string
	}{
		{
			buildOptions: types.ImageBuildOptions{
				SuppressOutput: true,
				NoCache:        true,
				Remove:         true,
				ForceRemove:    true,
				PullParent:     true,
			},
			expectedQueryParams: map[string]string{
				"q":       "1",
				"nocache": "1",
				"rm":      "1",
				"forcerm": "1",
				"pull":    "1",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: emptyRegistryConfig,
		},
		{
			buildOptions: types.ImageBuildOptions{
				SuppressOutput: false,
				NoCache:        false,
				Remove:         false,
				ForceRemove:    false,
				PullParent:     false,
			},
			expectedQueryParams: map[string]string{
				"q":       "",
				"nocache": "",
				"rm":      "0",
				"forcerm": "",
				"pull":    "",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: emptyRegistryConfig,
		},
		{
			buildOptions: types.ImageBuildOptions{
				RemoteContext: "remoteContext",
				Isolation:     container.Isolation("isolation"),
				CPUSetCPUs:    "2",
				CPUSetMems:    "12",
				CPUShares:     20,
				CPUQuota:      10,
				CPUPeriod:     30,
				Memory:        256,
				MemorySwap:    512,
				ShmSize:       10,
				CgroupParent:  "cgroup_parent",
				Dockerfile:    "Dockerfile",
			},
			expectedQueryParams: map[string]string{
				"remote":       "remoteContext",
				"isolation":    "isolation",
				"cpusetcpus":   "2",
				"cpusetmems":   "12",
				"cpushares":    "20",
				"cpuquota":     "10",
				"cpuperiod":    "30",
				"memory":       "256",
				"memswap":      "512",
				"shmsize":      "10",
				"cgroupparent": "cgroup_parent",
				"dockerfile":   "Dockerfile",
				"rm":           "0",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: emptyRegistryConfig,
		},
		{
			buildOptions: types.ImageBuildOptions{
				BuildArgs: map[string]*string{
					"ARG1": &v1,
					"ARG2": &v2,
					"ARG3": nil,
				},
			},
			expectedQueryParams: map[string]string{
				"buildargs": `{"ARG1":"value1","ARG2":"value2","ARG3":null}`,
				"rm":        "0",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: emptyRegistryConfig,
		},
		{
			buildOptions: types.ImageBuildOptions{
				Ulimits: []*units.Ulimit{
					{
						Name: "nproc",
						Hard: 65557,
						Soft: 65557,
					},
					{
						Name: "nofile",
						Hard: 20000,
						Soft: 40000,
					},
				},
			},
			expectedQueryParams: map[string]string{
				"ulimits": `[{"Name":"nproc","Hard":65557,"Soft":65557},{"Name":"nofile","Hard":20000,"Soft":40000}]`,
				"rm":      "0",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: emptyRegistryConfig,
		},
		{
			buildOptions: types.ImageBuildOptions{
				AuthConfigs: map[string]types.AuthConfig{
					"https://index.docker.io/v1/": {
						Auth: "dG90bwo=",
					},
				},
			},
			expectedQueryParams: map[string]string{
				"rm": "0",
			},
			expectedTags:           []string{},
			expectedRegistryConfig: "eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsiYXV0aCI6ImRHOTBid289In19",
		},
	}
	for _, buildCase := range buildCases {
		expectedURL := "/build"
		client := &Client{
			client: newMockClient(func(r *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(r.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, r.URL)
				}
				// Check request headers
				registryConfig := r.Header.Get("X-Registry-Config")
				if registryConfig != buildCase.expectedRegistryConfig {
					return nil, fmt.Errorf("X-Registry-Config header not properly set in the request. Expected '%s', got %s", buildCase.expectedRegistryConfig, registryConfig)
				}
				contentType := r.Header.Get("Content-Type")
				if contentType != "application/x-tar" {
					return nil, fmt.Errorf("Content-type header not properly set in the request. Expected 'application/x-tar', got %s", contentType)
				}

				// Check query parameters
				query := r.URL.Query()
				for key, expected := range buildCase.expectedQueryParams {
					actual := query.Get(key)
					if actual != expected {
						return nil, fmt.Errorf("%s not set in URL query properly. Expected '%s', got %s", key, expected, actual)
					}
				}

				// Check tags
				if len(buildCase.expectedTags) > 0 {
					tags := query["t"]
					if !reflect.DeepEqual(tags, buildCase.expectedTags) {
						return nil, fmt.Errorf("t (tags) not set in URL query properly. Expected '%s', got %s", buildCase.expectedTags, tags)
					}
				}

				headers := http.Header{}
				headers.Add("Server", "Docker/v1.23 (MyOS)")
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte("body"))),
					Header:     headers,
				}, nil
			}),
		}
		buildResponse, err := client.ImageBuild(context.Background(), nil, buildCase.buildOptions)
		if err != nil {
			t.Fatal(err)
		}
		if buildResponse.OSType != "MyOS" {
			t.Fatalf("expected OSType to be 'MyOS', got %s", buildResponse.OSType)
		}
		response, err := ioutil.ReadAll(buildResponse.Body)
		if err != nil {
			t.Fatal(err)
		}
		buildResponse.Body.Close()
		if string(response) != "body" {
			t.Fatalf("expected Body to contain 'body' string, got %s", response)
		}
	}
}

func TestGetDockerOS(t *testing.T) {
	cases := map[string]string{
		"Docker/v1.22 (linux)":   "linux",
		"Docker/v1.22 (windows)": "windows",
		"Foo/v1.22 (bar)":        "",
	}
	for header, os := range cases {
		g := getDockerOS(header)
		if g != os {
			t.Fatalf("Expected %s, got %s", os, g)
		}
	}
}
