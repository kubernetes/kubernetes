package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/docker/docker-credential-helpers/credentials"
)

// isValidCredsMessage checks if 'msg' contains invalid credentials error message.
// It returns whether the logs are free of invalid credentials errors and the error if it isn't.
// error values can be errCredentialsMissingServerURL or errCredentialsMissingUsername.
func isValidCredsMessage(msg string) error {
	if credentials.IsCredentialsMissingServerURLMessage(msg) {
		return credentials.NewErrCredentialsMissingServerURL()
	}
	if credentials.IsCredentialsMissingUsernameMessage(msg) {
		return credentials.NewErrCredentialsMissingUsername()
	}
	return nil
}

// Store uses an external program to save credentials.
func Store(program ProgramFunc, creds *credentials.Credentials) error {
	cmd := program(credentials.ActionStore)

	buffer := new(bytes.Buffer)
	if err := json.NewEncoder(buffer).Encode(creds); err != nil {
		return err
	}
	cmd.Input(buffer)

	out, err := cmd.Output()
	if err != nil {
		if isValidErr := isValidCredsMessage(string(out)); isValidErr != nil {
			err = isValidErr
		}
		return fmt.Errorf("error storing credentials - err: %v, out: `%s`", err, strings.TrimSpace(string(out)))
	}

	return nil
}

// Get executes an external program to get the credentials from a native store.
func Get(program ProgramFunc, serverURL string) (*credentials.Credentials, error) {
	cmd := program(credentials.ActionGet)
	cmd.Input(strings.NewReader(serverURL))

	out, err := cmd.Output()
	if err != nil {
		if credentials.IsErrCredentialsNotFoundMessage(string(out)) {
			return nil, credentials.NewErrCredentialsNotFound()
		}

		if isValidErr := isValidCredsMessage(string(out)); isValidErr != nil {
			err = isValidErr
		}

		return nil, fmt.Errorf("error getting credentials - err: %v, out: `%s`", err, strings.TrimSpace(string(out)))
	}

	resp := &credentials.Credentials{
		ServerURL: serverURL,
	}

	if err := json.NewDecoder(bytes.NewReader(out)).Decode(resp); err != nil {
		return nil, err
	}

	return resp, nil
}

// Erase executes a program to remove the server credentials from the native store.
func Erase(program ProgramFunc, serverURL string) error {
	cmd := program(credentials.ActionErase)
	cmd.Input(strings.NewReader(serverURL))
	out, err := cmd.Output()
	if err != nil {
		t := strings.TrimSpace(string(out))

		if isValidErr := isValidCredsMessage(t); isValidErr != nil {
			err = isValidErr
		}

		return fmt.Errorf("error erasing credentials - err: %v, out: `%s`", err, t)
	}

	return nil
}

// List executes a program to list server credentials in the native store.
func List(program ProgramFunc) (map[string]string, error) {
	cmd := program(credentials.ActionList)
	cmd.Input(strings.NewReader("unused"))
	out, err := cmd.Output()
	if err != nil {
		t := strings.TrimSpace(string(out))

		if isValidErr := isValidCredsMessage(t); isValidErr != nil {
			err = isValidErr
		}

		return nil, fmt.Errorf("error listing credentials - err: %v, out: `%s`", err, t)
	}

	var resp map[string]string
	if err = json.NewDecoder(bytes.NewReader(out)).Decode(&resp); err != nil {
		return nil, err
	}

	return resp, nil
}
