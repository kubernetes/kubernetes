package cleanhttp

import (
	"net/http"
	"strings"
	"unicode"
)

// HandlerInput provides input options to cleanhttp's handlers
type HandlerInput struct {
	ErrStatus int
}

// PrintablePathCheckHandler is a middleware that ensures the request path
// contains only printable runes.
func PrintablePathCheckHandler(next http.Handler, input *HandlerInput) http.Handler {
	// Nil-check on input to make it optional
	if input == nil {
		input = &HandlerInput{
			ErrStatus: http.StatusBadRequest,
		}
	}

	// Default to http.StatusBadRequest on error
	if input.ErrStatus == 0 {
		input.ErrStatus = http.StatusBadRequest
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check URL path for non-printable characters
		idx := strings.IndexFunc(r.URL.Path, func(c rune) bool {
			return !unicode.IsPrint(c)
		})

		if idx != -1 {
			w.WriteHeader(input.ErrStatus)
			return
		}

		next.ServeHTTP(w, r)
		return
	})
}
