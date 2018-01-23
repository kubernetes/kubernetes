package http

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

func WriteError(w http.ResponseWriter, code int, msg string) {
	e := struct {
		Error string `json:"error"`
	}{
		Error: msg,
	}
	b, err := json.Marshal(e)
	if err != nil {
		log.Printf("go-oidc: failed to marshal %#v: %v", e, err)
		code = http.StatusInternalServerError
		b = []byte(`{"error":"server_error"}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(b)
}

// BasicAuth parses a username and password from the request's
// Authorization header. This was pulled from golang master:
// https://codereview.appspot.com/76540043
func BasicAuth(r *http.Request) (username, password string, ok bool) {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return
	}

	if !strings.HasPrefix(auth, "Basic ") {
		return
	}
	c, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(auth, "Basic "))
	if err != nil {
		return
	}
	cs := string(c)
	s := strings.IndexByte(cs, ':')
	if s < 0 {
		return
	}
	return cs[:s], cs[s+1:], true
}

func cacheControlMaxAge(hdr string) (time.Duration, bool, error) {
	for _, field := range strings.Split(hdr, ",") {
		parts := strings.SplitN(strings.TrimSpace(field), "=", 2)
		k := strings.ToLower(strings.TrimSpace(parts[0]))
		if k != "max-age" {
			continue
		}

		if len(parts) == 1 {
			return 0, false, errors.New("max-age has no value")
		}

		v := strings.TrimSpace(parts[1])
		if v == "" {
			return 0, false, errors.New("max-age has empty value")
		}

		age, err := strconv.Atoi(v)
		if err != nil {
			return 0, false, err
		}

		if age <= 0 {
			return 0, false, nil
		}

		return time.Duration(age) * time.Second, true, nil
	}

	return 0, false, nil
}

func expires(date, expires string) (time.Duration, bool, error) {
	if date == "" || expires == "" {
		return 0, false, nil
	}

	te, err := time.Parse(time.RFC1123, expires)
	if err != nil {
		return 0, false, err
	}

	td, err := time.Parse(time.RFC1123, date)
	if err != nil {
		return 0, false, err
	}

	ttl := te.Sub(td)

	// headers indicate data already expired, caller should not
	// have to care about this case
	if ttl <= 0 {
		return 0, false, nil
	}

	return ttl, true, nil
}

func Cacheable(hdr http.Header) (time.Duration, bool, error) {
	ttl, ok, err := cacheControlMaxAge(hdr.Get("Cache-Control"))
	if err != nil || ok {
		return ttl, ok, err
	}

	return expires(hdr.Get("Date"), hdr.Get("Expires"))
}

// MergeQuery appends additional query values to an existing URL.
func MergeQuery(u url.URL, q url.Values) url.URL {
	uv := u.Query()
	for k, vs := range q {
		for _, v := range vs {
			uv.Add(k, v)
		}
	}
	u.RawQuery = uv.Encode()
	return u
}

// NewResourceLocation appends a resource id to the end of the requested URL path.
func NewResourceLocation(reqURL *url.URL, id string) string {
	var u url.URL
	u = *reqURL
	u.Path = path.Join(u.Path, id)
	u.RawQuery = ""
	u.Fragment = ""
	return u.String()
}

// CopyRequest returns a clone of the provided *http.Request.
// The returned object is a shallow copy of the struct and a
// deep copy of its Header field.
func CopyRequest(r *http.Request) *http.Request {
	r2 := *r
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return &r2
}
