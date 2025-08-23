package credentials

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// Action defines the name of an action (sub-command) supported by a
// credential-helper binary. It is an alias for "string", and mostly
// for convenience.
type Action = string

// List of actions (sub-commands) supported by credential-helper binaries.
const (
	ActionStore   Action = "store"
	ActionGet     Action = "get"
	ActionErase   Action = "erase"
	ActionList    Action = "list"
	ActionVersion Action = "version"
)

// Credentials holds the information shared between docker and the credentials store.
type Credentials struct {
	ServerURL string
	Username  string
	Secret    string
}

// isValid checks the integrity of Credentials object such that no credentials lack
// a server URL or a username.
// It returns whether the credentials are valid and the error if it isn't.
// error values can be errCredentialsMissingServerURL or errCredentialsMissingUsername
func (c *Credentials) isValid() (bool, error) {
	if len(c.ServerURL) == 0 {
		return false, NewErrCredentialsMissingServerURL()
	}

	if len(c.Username) == 0 {
		return false, NewErrCredentialsMissingUsername()
	}

	return true, nil
}

// CredsLabel holds the way Docker credentials should be labeled as such in credentials stores that allow labelling.
// That label allows to filter out non-Docker credentials too at lookup/search in macOS keychain,
// Windows credentials manager and Linux libsecret. Default value is "Docker Credentials"
var CredsLabel = "Docker Credentials"

// SetCredsLabel is a simple setter for CredsLabel
func SetCredsLabel(label string) {
	CredsLabel = label
}

// Serve initializes the credentials-helper and parses the action argument.
// This function is designed to be called from a command line interface.
// It uses os.Args[1] as the key for the action.
// It uses os.Stdin as input and os.Stdout as output.
// This function terminates the program with os.Exit(1) if there is an error.
func Serve(helper Helper) {
	if len(os.Args) != 2 {
		_, _ = fmt.Fprintln(os.Stdout, usage())
		os.Exit(1)
	}

	switch os.Args[1] {
	case "--version", "-v":
		_ = PrintVersion(os.Stdout)
		os.Exit(0)
	case "--help", "-h":
		_, _ = fmt.Fprintln(os.Stdout, usage())
		os.Exit(0)
	}

	if err := HandleCommand(helper, os.Args[1], os.Stdin, os.Stdout); err != nil {
		_, _ = fmt.Fprintln(os.Stdout, err)
		os.Exit(1)
	}
}

func usage() string {
	return fmt.Sprintf("Usage: %s <store|get|erase|list|version>", Name)
}

// HandleCommand runs a helper to execute a credential action.
func HandleCommand(helper Helper, action Action, in io.Reader, out io.Writer) error {
	switch action {
	case ActionStore:
		return Store(helper, in)
	case ActionGet:
		return Get(helper, in, out)
	case ActionErase:
		return Erase(helper, in)
	case ActionList:
		return List(helper, out)
	case ActionVersion:
		return PrintVersion(out)
	default:
		return fmt.Errorf("%s: unknown action: %s", Name, action)
	}
}

// Store uses a helper and an input reader to save credentials.
// The reader must contain the JSON serialization of a Credentials struct.
func Store(helper Helper, reader io.Reader) error {
	scanner := bufio.NewScanner(reader)

	buffer := new(bytes.Buffer)
	for scanner.Scan() {
		buffer.Write(scanner.Bytes())
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}

	var creds Credentials
	if err := json.NewDecoder(buffer).Decode(&creds); err != nil {
		return err
	}

	if ok, err := creds.isValid(); !ok {
		return err
	}

	return helper.Add(&creds)
}

// Get retrieves the credentials for a given server url.
// The reader must contain the server URL to search.
// The writer is used to write the JSON serialization of the credentials.
func Get(helper Helper, reader io.Reader, writer io.Writer) error {
	scanner := bufio.NewScanner(reader)

	buffer := new(bytes.Buffer)
	for scanner.Scan() {
		buffer.Write(scanner.Bytes())
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}

	serverURL := strings.TrimSpace(buffer.String())
	if len(serverURL) == 0 {
		return NewErrCredentialsMissingServerURL()
	}

	username, secret, err := helper.Get(serverURL)
	if err != nil {
		return err
	}

	buffer.Reset()
	err = json.NewEncoder(buffer).Encode(Credentials{
		ServerURL: serverURL,
		Username:  username,
		Secret:    secret,
	})
	if err != nil {
		return err
	}

	_, _ = fmt.Fprint(writer, buffer.String())
	return nil
}

// Erase removes credentials from the store.
// The reader must contain the server URL to remove.
func Erase(helper Helper, reader io.Reader) error {
	scanner := bufio.NewScanner(reader)

	buffer := new(bytes.Buffer)
	for scanner.Scan() {
		buffer.Write(scanner.Bytes())
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}

	serverURL := strings.TrimSpace(buffer.String())
	if len(serverURL) == 0 {
		return NewErrCredentialsMissingServerURL()
	}

	return helper.Delete(serverURL)
}

// List returns all the serverURLs of keys in
// the OS store as a list of strings
func List(helper Helper, writer io.Writer) error {
	accts, err := helper.List()
	if err != nil {
		return err
	}
	return json.NewEncoder(writer).Encode(accts)
}

// PrintVersion outputs the current version.
func PrintVersion(writer io.Writer) error {
	_, _ = fmt.Fprintf(writer, "%s (%s) %s\n", Name, Package, Version)
	return nil
}
