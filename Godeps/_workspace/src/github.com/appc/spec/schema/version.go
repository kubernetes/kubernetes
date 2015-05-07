package schema

import (
	"github.com/appc/spec/schema/types"
)

const (
	// version represents the canonical version of the appc spec and tooling.
	// For now, the schema and tooling is coupled with the spec itself, so
	// this must be kept in sync with the VERSION file in the root of the repo.
	version string = "0.5.1+git"
)

var (
	// AppContainerVersion is the SemVer representation of version
	AppContainerVersion types.SemVer
)

func init() {
	v, err := types.NewSemVer(version)
	if err != nil {
		panic(err)
	}
	AppContainerVersion = *v
}
