package image

import (
	"encoding/json"
	"fmt"
	"regexp"
	"time"

	"github.com/docker/docker/runconfig"
)

var validHex = regexp.MustCompile(`^([a-f0-9]{64})$`)

type Image struct {
	ID              string            `json:"id"`
	Parent          string            `json:"parent,omitempty"`
	Comment         string            `json:"comment,omitempty"`
	Created         time.Time         `json:"created"`
	Container       string            `json:"container,omitempty"`
	ContainerConfig runconfig.Config  `json:"container_config,omitempty"`
	DockerVersion   string            `json:"docker_version,omitempty"`
	Author          string            `json:"author,omitempty"`
	Config          *runconfig.Config `json:"config,omitempty"`
	Architecture    string            `json:"architecture,omitempty"`
	OS              string            `json:"os,omitempty"`
	Size            int64
}

// Build an Image object from raw json data
func NewImgJSON(src []byte) (*Image, error) {
	ret := &Image{}

	// FIXME: Is there a cleaner way to "purify" the input json?
	if err := json.Unmarshal(src, ret); err != nil {
		return nil, err
	}
	return ret, nil
}

// Check wheather id is a valid image ID or not
func ValidateID(id string) error {
	if ok := validHex.MatchString(id); !ok {
		return fmt.Errorf("image ID '%s' is invalid", id)
	}
	return nil
}
