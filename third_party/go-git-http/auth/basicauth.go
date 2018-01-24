package auth

import (
	"encoding/base64"
	"fmt"
	"regexp"
	"strings"
)

// Parse http basic header
type BasicAuth struct {
	Name string
	Pass string
}

var (
	basicAuthRegex = regexp.MustCompile("^([^:]*):(.*)$")
)

func parseAuthHeader(header string) (*BasicAuth, error) {
	parts := strings.SplitN(header, " ", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("Invalid authorization header, not enought parts")
	}

	authType := parts[0]
	authData := parts[1]

	if strings.ToLower(authType) != "basic" {
		return nil, fmt.Errorf("Authentication '%s' was not of 'Basic' type", authType)
	}

	data, err := base64.StdEncoding.DecodeString(authData)
	if err != nil {
		return nil, err
	}

	matches := basicAuthRegex.FindStringSubmatch(string(data))
	if matches == nil {
		return nil, fmt.Errorf("Authorization data '%s' did not match auth regexp", data)
	}

	return &BasicAuth{
		Name: matches[1],
		Pass: matches[2],
	}, nil
}
