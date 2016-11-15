package reference

import (
	distreference "github.com/docker/distribution/reference"
)

// Parse parses the given references and returns the repository and
// tag (if present) from it. If there is an error during parsing, it will
// return an error.
func Parse(ref string) (string, string, error) {
	distributionRef, err := distreference.ParseNamed(ref)
	if err != nil {
		return "", "", err
	}

	tag := GetTagFromNamedRef(distributionRef)
	return distributionRef.Name(), tag, nil
}

// GetTagFromNamedRef returns a tag from the specified reference.
// This function is necessary as long as the docker "server" api makes the distinction between repository
// and tags.
func GetTagFromNamedRef(ref distreference.Named) string {
	var tag string
	switch x := ref.(type) {
	case distreference.Digested:
		tag = x.Digest().String()
	case distreference.NamedTagged:
		tag = x.Tag()
	default:
		tag = "latest"
	}
	return tag
}
