package storageos

import (
	"encoding/json"
	"errors"
)

var (
	// LoginAPIPrefix is a partial path to the HTTP endpoint.
	LoginAPIPrefix = "auth/login"
	// ErrLoginFailed is the error returned on an unsuccessful login.
	ErrLoginFailed = errors.New("failed to get token from API endpoint")
)

// Login attemps to get a token from the API
func (c *Client) Login() (token string, err error) {
	resp, err := c.do("POST", LoginAPIPrefix, doOptions{data: struct {
		User string `json:"username"`
		Pass string `json:"password"`
	}{c.username, c.secret}})

	if err != nil {
		if _, ok := err.(*Error); ok {
			return "", ErrLoginFailed
		}

		return "", err
	}

	if resp.StatusCode != 200 {
		return "", ErrLoginFailed
	}

	unmarsh := struct {
		Token string `json:"token"`
	}{}

	if err := json.NewDecoder(resp.Body).Decode(&unmarsh); err != nil {
		return "", err
	}

	if unmarsh.Token == "" {
		return "", ErrLoginFailed
	}

	return unmarsh.Token, nil
}
