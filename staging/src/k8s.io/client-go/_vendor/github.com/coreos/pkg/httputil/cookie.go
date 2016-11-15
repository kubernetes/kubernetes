package httputil

import (
	"net/http"
	"time"
)

// DeleteCookies effectively deletes all named cookies
// by wiping all data and setting to expire immediately.
func DeleteCookies(w http.ResponseWriter, cookieNames ...string) {
	for _, n := range cookieNames {
		c := &http.Cookie{
			Name:    n,
			Value:   "",
			Path:    "/",
			MaxAge:  -1,
			Expires: time.Time{},
		}
		http.SetCookie(w, c)
	}
}
