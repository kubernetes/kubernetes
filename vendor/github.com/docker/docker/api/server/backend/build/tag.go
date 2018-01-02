package build

import (
	"fmt"
	"io"
	"runtime"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/system"
	"github.com/pkg/errors"
)

// Tagger is responsible for tagging an image created by a builder
type Tagger struct {
	imageComponent ImageComponent
	stdout         io.Writer
	repoAndTags    []reference.Named
}

// NewTagger returns a new Tagger for tagging the images of a build.
// If any of the names are invalid tags an error is returned.
func NewTagger(backend ImageComponent, stdout io.Writer, names []string) (*Tagger, error) {
	reposAndTags, err := sanitizeRepoAndTags(names)
	if err != nil {
		return nil, err
	}
	return &Tagger{
		imageComponent: backend,
		stdout:         stdout,
		repoAndTags:    reposAndTags,
	}, nil
}

// TagImages creates image tags for the imageID
func (bt *Tagger) TagImages(imageID image.ID) error {
	for _, rt := range bt.repoAndTags {
		// TODO @jhowardmsft LCOW support. Will need revisiting.
		platform := runtime.GOOS
		if system.LCOWSupported() {
			platform = "linux"
		}
		if err := bt.imageComponent.TagImageWithReference(imageID, platform, rt); err != nil {
			return err
		}
		fmt.Fprintf(bt.stdout, "Successfully tagged %s\n", reference.FamiliarString(rt))
	}
	return nil
}

// sanitizeRepoAndTags parses the raw "t" parameter received from the client
// to a slice of repoAndTag.
// It also validates each repoName and tag.
func sanitizeRepoAndTags(names []string) ([]reference.Named, error) {
	var (
		repoAndTags []reference.Named
		// This map is used for deduplicating the "-t" parameter.
		uniqNames = make(map[string]struct{})
	)
	for _, repo := range names {
		if repo == "" {
			continue
		}

		ref, err := reference.ParseNormalizedNamed(repo)
		if err != nil {
			return nil, err
		}

		if _, isCanonical := ref.(reference.Canonical); isCanonical {
			return nil, errors.New("build tag cannot contain a digest")
		}

		ref = reference.TagNameOnly(ref)

		nameWithTag := ref.String()

		if _, exists := uniqNames[nameWithTag]; !exists {
			uniqNames[nameWithTag] = struct{}{}
			repoAndTags = append(repoAndTags, ref)
		}
	}
	return repoAndTags, nil
}
