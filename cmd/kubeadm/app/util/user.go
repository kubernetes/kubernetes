package util

import "os/user"

const (
	sudoUser = "0"
)

func IsRoot() (*user.User, bool) {
	u, err := user.Current()
	if err != nil || u.Uid != sudoUser {
		return u, false
	}
	return u, true
}
