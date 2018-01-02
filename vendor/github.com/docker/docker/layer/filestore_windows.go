package layer

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

// SetPlatform writes the "platform" file to the layer filestore
func (fm *fileMetadataTransaction) SetPlatform(platform Platform) error {
	if platform == "" {
		return nil
	}
	return fm.ws.WriteFile("platform", []byte(platform), 0644)
}

// GetPlatform reads the "platform" file from the layer filestore
func (fms *fileMetadataStore) GetPlatform(layer ChainID) (Platform, error) {
	contentBytes, err := ioutil.ReadFile(fms.getLayerFilename(layer, "platform"))
	if err != nil {
		// For backwards compatibility, the platform file may not exist. Default to "windows" if missing.
		if os.IsNotExist(err) {
			return "windows", nil
		}
		return "", err
	}
	content := strings.TrimSpace(string(contentBytes))

	if content != "windows" && content != "linux" {
		return "", fmt.Errorf("invalid platform value: %s", content)
	}

	return Platform(content), nil
}
