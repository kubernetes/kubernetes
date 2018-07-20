package httputils

import (
	"net/http"
	"strconv"
	"strings"
)

// BoolValue transforms a form value in different formats into a boolean type.
func BoolValue(r *http.Request, k string) bool {
	s := strings.ToLower(strings.TrimSpace(r.FormValue(k)))
	return !(s == "" || s == "0" || s == "no" || s == "false" || s == "none")
}

// BoolValueOrDefault returns the default bool passed if the query param is
// missing, otherwise it's just a proxy to boolValue above.
func BoolValueOrDefault(r *http.Request, k string, d bool) bool {
	if _, ok := r.Form[k]; !ok {
		return d
	}
	return BoolValue(r, k)
}

// Int64ValueOrZero parses a form value into an int64 type.
// It returns 0 if the parsing fails.
func Int64ValueOrZero(r *http.Request, k string) int64 {
	val, err := Int64ValueOrDefault(r, k, 0)
	if err != nil {
		return 0
	}
	return val
}

// Int64ValueOrDefault parses a form value into an int64 type. If there is an
// error, returns the error. If there is no value returns the default value.
func Int64ValueOrDefault(r *http.Request, field string, def int64) (int64, error) {
	if r.Form.Get(field) != "" {
		value, err := strconv.ParseInt(r.Form.Get(field), 10, 64)
		return value, err
	}
	return def, nil
}

// ArchiveOptions stores archive information for different operations.
type ArchiveOptions struct {
	Name string
	Path string
}

type badParameterError struct {
	param string
}

func (e badParameterError) Error() string {
	return "bad parameter: " + e.param + "cannot be empty"
}

func (e badParameterError) InvalidParameter() {}

// ArchiveFormValues parses form values and turns them into ArchiveOptions.
// It fails if the archive name and path are not in the request.
func ArchiveFormValues(r *http.Request, vars map[string]string) (ArchiveOptions, error) {
	if err := ParseForm(r); err != nil {
		return ArchiveOptions{}, err
	}

	name := vars["name"]
	if name == "" {
		return ArchiveOptions{}, badParameterError{"name"}
	}
	path := r.Form.Get("path")
	if path == "" {
		return ArchiveOptions{}, badParameterError{"path"}
	}
	return ArchiveOptions{name, path}, nil
}
