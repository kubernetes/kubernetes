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
	"golang.org/x/net/context"
)

func TestContainerCommitError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerCommit(context.Background(), "nothing", types.ContainerCommitOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerCommit(t *testing.T) {
	expectedURL := "/commit"
	expectedContainerID := "container_id"
	specifiedReference := "repository_name:tag"
	expectedRepositoryName := "repository_name"
	expectedTag := "tag"
	expectedComment := "comment"
	expectedAuthor := "author"
	expectedChanges := []string{"change1", "change2"}

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			query := req.URL.Query()
			containerID := query.Get("container")
			if containerID != expectedContainerID {
				return nil, fmt.Errorf("container id not set in URL query properly. Expected '%s', got %s", expectedContainerID, containerID)
			}
			repo := query.Get("repo")
			if repo != expectedRepositoryName {
				return nil, fmt.Errorf("container repo not set in URL query properly. Expected '%s', got %s", expectedRepositoryName, repo)
			}
			tag := query.Get("tag")
			if tag != expectedTag {
				return nil, fmt.Errorf("container tag not set in URL query properly. Expected '%s', got %s'", expectedTag, tag)
			}
			comment := query.Get("comment")
			if comment != expectedComment {
				return nil, fmt.Errorf("container comment not set in URL query properly. Expected '%s', got %s'", expectedComment, comment)
			}
			author := query.Get("author")
			if author != expectedAuthor {
				return nil, fmt.Errorf("container author not set in URL query properly. Expected '%s', got %s'", expectedAuthor, author)
			}
			pause := query.Get("pause")
			if pause != "0" {
				return nil, fmt.Errorf("container pause not set in URL query properly. Expected 'true', got %v'", pause)
			}
			changes := query["changes"]
			if len(changes) != len(expectedChanges) {
				return nil, fmt.Errorf("expected container changes size to be '%d', got %d", len(expectedChanges), len(changes))
			}
			b, err := json.Marshal(types.IDResponse{
				ID: "new_container_id",
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

	r, err := client.ContainerCommit(context.Background(), expectedContainerID, types.ContainerCommitOptions{
		Reference: specifiedReference,
		Comment:   expectedComment,
		Author:    expectedAuthor,
		Changes:   expectedChanges,
		Pause:     false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "new_container_id" {
		t.Fatalf("expected `new_container_id`, got %s", r.ID)
	}
}
