// +build !windows

package layer

// SetPlatform writes the "platform" file to the layer filestore
func (fm *fileMetadataTransaction) SetPlatform(platform Platform) error {
	return nil
}

// GetPlatform reads the "platform" file from the layer filestore
func (fms *fileMetadataStore) GetPlatform(layer ChainID) (Platform, error) {
	return "", nil
}
