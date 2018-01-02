package v1

import (
	"encoding/json"
	"reflect"
	"strings"

	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/stringid"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
)

// noFallbackMinVersion is the minimum version for which v1compatibility
// information will not be marshaled through the Image struct to remove
// blank fields.
var noFallbackMinVersion = "1.8.3"

// HistoryFromConfig creates a History struct from v1 configuration JSON
func HistoryFromConfig(imageJSON []byte, emptyLayer bool) (image.History, error) {
	h := image.History{}
	var v1Image image.V1Image
	if err := json.Unmarshal(imageJSON, &v1Image); err != nil {
		return h, err
	}

	return image.History{
		Author:     v1Image.Author,
		Created:    v1Image.Created,
		CreatedBy:  strings.Join(v1Image.ContainerConfig.Cmd, " "),
		Comment:    v1Image.Comment,
		EmptyLayer: emptyLayer,
	}, nil
}

// CreateID creates an ID from v1 image, layerID and parent ID.
// Used for backwards compatibility with old clients.
func CreateID(v1Image image.V1Image, layerID layer.ChainID, parent digest.Digest) (digest.Digest, error) {
	v1Image.ID = ""
	v1JSON, err := json.Marshal(v1Image)
	if err != nil {
		return "", err
	}

	var config map[string]*json.RawMessage
	if err := json.Unmarshal(v1JSON, &config); err != nil {
		return "", err
	}

	// FIXME: note that this is slightly incompatible with RootFS logic
	config["layer_id"] = rawJSON(layerID)
	if parent != "" {
		config["parent"] = rawJSON(parent)
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return "", err
	}
	logrus.Debugf("CreateV1ID %s", configJSON)

	return digest.FromBytes(configJSON), nil
}

// MakeConfigFromV1Config creates an image config from the legacy V1 config format.
func MakeConfigFromV1Config(imageJSON []byte, rootfs *image.RootFS, history []image.History) ([]byte, error) {
	var dver struct {
		DockerVersion string `json:"docker_version"`
	}

	if err := json.Unmarshal(imageJSON, &dver); err != nil {
		return nil, err
	}

	useFallback := versions.LessThan(dver.DockerVersion, noFallbackMinVersion)

	if useFallback {
		var v1Image image.V1Image
		err := json.Unmarshal(imageJSON, &v1Image)
		if err != nil {
			return nil, err
		}
		imageJSON, err = json.Marshal(v1Image)
		if err != nil {
			return nil, err
		}
	}

	var c map[string]*json.RawMessage
	if err := json.Unmarshal(imageJSON, &c); err != nil {
		return nil, err
	}

	delete(c, "id")
	delete(c, "parent")
	delete(c, "Size") // Size is calculated from data on disk and is inconsistent
	delete(c, "parent_id")
	delete(c, "layer_id")
	delete(c, "throwaway")

	c["rootfs"] = rawJSON(rootfs)
	c["history"] = rawJSON(history)

	return json.Marshal(c)
}

// MakeV1ConfigFromConfig creates a legacy V1 image config from an Image struct
func MakeV1ConfigFromConfig(img *image.Image, v1ID, parentV1ID string, throwaway bool) ([]byte, error) {
	// Top-level v1compatibility string should be a modified version of the
	// image config.
	var configAsMap map[string]*json.RawMessage
	if err := json.Unmarshal(img.RawJSON(), &configAsMap); err != nil {
		return nil, err
	}

	// Delete fields that didn't exist in old manifest
	imageType := reflect.TypeOf(img).Elem()
	for i := 0; i < imageType.NumField(); i++ {
		f := imageType.Field(i)
		jsonName := strings.Split(f.Tag.Get("json"), ",")[0]
		// Parent is handled specially below.
		if jsonName != "" && jsonName != "parent" {
			delete(configAsMap, jsonName)
		}
	}
	configAsMap["id"] = rawJSON(v1ID)
	if parentV1ID != "" {
		configAsMap["parent"] = rawJSON(parentV1ID)
	}
	if throwaway {
		configAsMap["throwaway"] = rawJSON(true)
	}

	return json.Marshal(configAsMap)
}

func rawJSON(value interface{}) *json.RawMessage {
	jsonval, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return (*json.RawMessage)(&jsonval)
}

// ValidateID checks whether an ID string is a valid image ID.
func ValidateID(id string) error {
	return stringid.ValidateID(id)
}
