package layer

import "github.com/docker/distribution"

var _ distribution.Describable = &roLayer{}

func (rl *roLayer) Descriptor() distribution.Descriptor {
	return rl.descriptor
}

func (rl *roLayer) Platform() Platform {
	if rl.platform == "" {
		return "windows"
	}
	return rl.platform
}
