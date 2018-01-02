package meta

import "golang.org/x/crypto/bcrypt"

func init() {
	bcryptCost = bcrypt.MinCost
}
