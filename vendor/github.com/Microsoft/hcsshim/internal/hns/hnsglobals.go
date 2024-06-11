//go:build windows

package hns

type HNSGlobals struct {
	Version HNSVersion `json:"Version"`
}

type HNSVersion struct {
	Major int `json:"Major"`
	Minor int `json:"Minor"`
}

var (
	HNSVersion1803 = HNSVersion{Major: 7, Minor: 2}
)

func GetHNSGlobals() (*HNSGlobals, error) {
	var version HNSVersion
	err := hnsCall("GET", "/globals/version", "", &version)
	if err != nil {
		return nil, err
	}

	globals := &HNSGlobals{
		Version: version,
	}

	return globals, nil
}
