package aetest

import (
	"hash/crc32"
	"net/http"
	"strconv"

	"google.golang.org/appengine/user"
)

// Login causes the provided Request to act as though issued by the given user.
func Login(u *user.User, req *http.Request) {
	req.Header.Set("X-AppEngine-User-Email", u.Email)
	id := u.ID
	if id == "" {
		id = strconv.Itoa(int(crc32.Checksum([]byte(u.Email), crc32.IEEETable)))
	}
	req.Header.Set("X-AppEngine-User-Id", id)
	req.Header.Set("X-AppEngine-User-Federated-Identity", u.Email)
	req.Header.Set("X-AppEngine-User-Federated-Provider", u.FederatedProvider)
	if u.Admin {
		req.Header.Set("X-AppEngine-User-Is-Admin", "1")
	} else {
		req.Header.Set("X-AppEngine-User-Is-Admin", "0")
	}
}

// Logout causes the provided Request to act as though issued by a logged-out
// user.
func Logout(req *http.Request) {
	req.Header.Del("X-AppEngine-User-Email")
	req.Header.Del("X-AppEngine-User-Id")
	req.Header.Del("X-AppEngine-User-Is-Admin")
	req.Header.Del("X-AppEngine-User-Federated-Identity")
	req.Header.Del("X-AppEngine-User-Federated-Provider")
}
