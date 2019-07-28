package wclayer

import "os"

// ProcessBaseLayer post-processes a base layer that has had its files extracted.
// The files should have been extracted to <path>\Files.
func ProcessBaseLayer(path string) error {
	err := processBaseImage(path)
	if err != nil {
		return &os.PathError{Op: "ProcessBaseLayer", Path: path, Err: err}
	}
	return nil
}

// ProcessUtilityVMImage post-processes a utility VM image that has had its files extracted.
// The files should have been extracted to <path>\Files.
func ProcessUtilityVMImage(path string) error {
	err := processUtilityImage(path)
	if err != nil {
		return &os.PathError{Op: "ProcessUtilityVMImage", Path: path, Err: err}
	}
	return nil
}
