// +build !windows

package layer

func (rl *roLayer) Platform() Platform {
	return ""
}
