package graph

import (
	"fmt"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/image"
	"github.com/docker/docker/utils"
)

// WalkHistory calls the handler function for each image in the
// provided images lineage starting from immediate parent.
func (graph *Graph) WalkHistory(img *image.Image, handler func(image.Image) error) (err error) {
	currentImg := img
	for currentImg != nil {
		if handler != nil {
			if err := handler(*currentImg); err != nil {
				return err
			}
		}
		currentImg, err = graph.GetParent(currentImg)
		if err != nil {
			return fmt.Errorf("Error while getting parent image: %v", err)
		}
	}
	return nil
}

// depth returns the number of parents for a
// current image
func (graph *Graph) depth(img *image.Image) (int, error) {
	var (
		count  = 0
		parent = img
		err    error
	)

	for parent != nil {
		count++
		parent, err = graph.GetParent(parent)
		if err != nil {
			return -1, err
		}
	}
	return count, nil
}

// Set the max depth to the aufs default that most
// kernels are compiled with
// For more information see: http://sourceforge.net/p/aufs/aufs3-standalone/ci/aufs3.12/tree/config.mk
const MaxImageDepth = 127

// CheckDepth returns an error if the depth of an image, as returned
// by ImageDepth, is too large to support creating a container from it
// on this daemon.
func (graph *Graph) CheckDepth(img *image.Image) error {
	// We add 2 layers to the depth because the container's rw and
	// init layer add to the restriction
	depth, err := graph.depth(img)
	if err != nil {
		return err
	}
	if depth+2 >= MaxImageDepth {
		return fmt.Errorf("Cannot create container with more than %d parents", MaxImageDepth)
	}
	return nil
}

func (s *TagStore) History(name string) ([]*types.ImageHistory, error) {
	foundImage, err := s.LookupImage(name)
	if err != nil {
		return nil, err
	}

	lookupMap := make(map[string][]string)
	for name, repository := range s.Repositories {
		for tag, id := range repository {
			// If the ID already has a reverse lookup, do not update it unless for "latest"
			if _, exists := lookupMap[id]; !exists {
				lookupMap[id] = []string{}
			}
			lookupMap[id] = append(lookupMap[id], utils.ImageReference(name, tag))
		}
	}

	history := []*types.ImageHistory{}

	err = s.graph.WalkHistory(foundImage, func(img image.Image) error {
		history = append(history, &types.ImageHistory{
			ID:        img.ID,
			Created:   img.Created.Unix(),
			CreatedBy: strings.Join(img.ContainerConfig.Cmd.Slice(), " "),
			Tags:      lookupMap[img.ID],
			Size:      img.Size,
			Comment:   img.Comment,
		})
		return nil
	})

	return history, err
}

func (graph *Graph) GetParent(img *image.Image) (*image.Image, error) {
	if img.Parent == "" {
		return nil, nil
	}
	return graph.Get(img.Parent)
}

func (graph *Graph) GetParentsSize(img *image.Image, size int64) int64 {
	parentImage, err := graph.GetParent(img)
	if err != nil || parentImage == nil {
		return size
	}
	size += parentImage.Size
	return graph.GetParentsSize(parentImage, size)
}
