package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	registrytypes "github.com/docker/docker/api/types/registry"
	"github.com/docker/docker/api/types/swarm"
	"github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestServiceCreateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ServiceCreate(context.Background(), swarm.ServiceSpec{}, types.ServiceCreateOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestServiceCreate(t *testing.T) {
	expectedURL := "/services/create"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}
			b, err := json.Marshal(types.ServiceCreateResponse{
				ID: "service_id",
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	r, err := client.ServiceCreate(context.Background(), swarm.ServiceSpec{}, types.ServiceCreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "service_id" {
		t.Fatalf("expected `service_id`, got %s", r.ID)
	}
}

func TestServiceCreateCompatiblePlatforms(t *testing.T) {
	client := &Client{
		version: "1.30",
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if strings.HasPrefix(req.URL.Path, "/v1.30/services/create") {
				var serviceSpec swarm.ServiceSpec

				// check if the /distribution endpoint returned correct output
				err := json.NewDecoder(req.Body).Decode(&serviceSpec)
				if err != nil {
					return nil, err
				}

				assert.Equal(t, "foobar:1.0@sha256:c0537ff6a5218ef531ece93d4984efc99bbf3f7497c0a7726c88e2bb7584dc96", serviceSpec.TaskTemplate.ContainerSpec.Image)
				assert.Len(t, serviceSpec.TaskTemplate.Placement.Platforms, 1)

				p := serviceSpec.TaskTemplate.Placement.Platforms[0]
				b, err := json.Marshal(types.ServiceCreateResponse{
					ID: "service_" + p.OS + "_" + p.Architecture,
				})
				if err != nil {
					return nil, err
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader(b)),
				}, nil
			} else if strings.HasPrefix(req.URL.Path, "/v1.30/distribution/") {
				b, err := json.Marshal(registrytypes.DistributionInspect{
					Descriptor: v1.Descriptor{
						Digest: "sha256:c0537ff6a5218ef531ece93d4984efc99bbf3f7497c0a7726c88e2bb7584dc96",
					},
					Platforms: []v1.Platform{
						{
							Architecture: "amd64",
							OS:           "linux",
						},
					},
				})
				if err != nil {
					return nil, err
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader(b)),
				}, nil
			} else {
				return nil, fmt.Errorf("unexpected URL '%s'", req.URL.Path)
			}
		}),
	}

	spec := swarm.ServiceSpec{TaskTemplate: swarm.TaskSpec{ContainerSpec: &swarm.ContainerSpec{Image: "foobar:1.0"}}}

	r, err := client.ServiceCreate(context.Background(), spec, types.ServiceCreateOptions{QueryRegistry: true})
	assert.NoError(t, err)
	assert.Equal(t, "service_linux_amd64", r.ID)
}

func TestServiceCreateDigestPinning(t *testing.T) {
	dgst := "sha256:c0537ff6a5218ef531ece93d4984efc99bbf3f7497c0a7726c88e2bb7584dc96"
	dgstAlt := "sha256:37ffbf3f7497c07584dc9637ffbf3f7497c0758c0537ffbf3f7497c0c88e2bb7"
	serviceCreateImage := ""
	pinByDigestTests := []struct {
		img      string // input image provided by the user
		expected string // expected image after digest pinning
	}{
		// default registry returns familiar string
		{"docker.io/library/alpine", "alpine:latest@" + dgst},
		// provided tag is preserved and digest added
		{"alpine:edge", "alpine:edge@" + dgst},
		// image with provided alternative digest remains unchanged
		{"alpine@" + dgstAlt, "alpine@" + dgstAlt},
		// image with provided tag and alternative digest remains unchanged
		{"alpine:edge@" + dgstAlt, "alpine:edge@" + dgstAlt},
		// image on alternative registry does not result in familiar string
		{"alternate.registry/library/alpine", "alternate.registry/library/alpine:latest@" + dgst},
		// unresolvable image does not get a digest
		{"cannotresolve", "cannotresolve:latest"},
	}

	client := &Client{
		version: "1.30",
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if strings.HasPrefix(req.URL.Path, "/v1.30/services/create") {
				// reset and set image received by the service create endpoint
				serviceCreateImage = ""
				var service swarm.ServiceSpec
				if err := json.NewDecoder(req.Body).Decode(&service); err != nil {
					return nil, fmt.Errorf("could not parse service create request")
				}
				serviceCreateImage = service.TaskTemplate.ContainerSpec.Image

				b, err := json.Marshal(types.ServiceCreateResponse{
					ID: "service_id",
				})
				if err != nil {
					return nil, err
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader(b)),
				}, nil
			} else if strings.HasPrefix(req.URL.Path, "/v1.30/distribution/cannotresolve") {
				// unresolvable image
				return nil, fmt.Errorf("cannot resolve image")
			} else if strings.HasPrefix(req.URL.Path, "/v1.30/distribution/") {
				// resolvable images
				b, err := json.Marshal(registrytypes.DistributionInspect{
					Descriptor: v1.Descriptor{
						Digest: digest.Digest(dgst),
					},
				})
				if err != nil {
					return nil, err
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader(b)),
				}, nil
			}
			return nil, fmt.Errorf("unexpected URL '%s'", req.URL.Path)
		}),
	}

	// run pin by digest tests
	for _, p := range pinByDigestTests {
		r, err := client.ServiceCreate(context.Background(), swarm.ServiceSpec{
			TaskTemplate: swarm.TaskSpec{
				ContainerSpec: &swarm.ContainerSpec{
					Image: p.img,
				},
			},
		}, types.ServiceCreateOptions{QueryRegistry: true})

		if err != nil {
			t.Fatal(err)
		}

		if r.ID != "service_id" {
			t.Fatalf("expected `service_id`, got %s", r.ID)
		}

		if p.expected != serviceCreateImage {
			t.Fatalf("expected image %s, got %s", p.expected, serviceCreateImage)
		}
	}
}
