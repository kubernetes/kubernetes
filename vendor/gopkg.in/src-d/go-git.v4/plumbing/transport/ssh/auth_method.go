package ssh

import (
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/user"
	"path/filepath"

	"gopkg.in/src-d/go-git.v4/plumbing/transport"

	"github.com/mitchellh/go-homedir"
	"github.com/xanzy/ssh-agent"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

const DefaultUsername = "git"

// AuthMethod is the interface all auth methods for the ssh client
// must implement. The clientConfig method returns the ssh client
// configuration needed to establish an ssh connection.
type AuthMethod interface {
	transport.AuthMethod
	clientConfig() *ssh.ClientConfig
	hostKeyCallback() (ssh.HostKeyCallback, error)
}

// The names of the AuthMethod implementations. To be returned by the
// Name() method. Most git servers only allow PublicKeysName and
// PublicKeysCallbackName.
const (
	KeyboardInteractiveName = "ssh-keyboard-interactive"
	PasswordName            = "ssh-password"
	PasswordCallbackName    = "ssh-password-callback"
	PublicKeysName          = "ssh-public-keys"
	PublicKeysCallbackName  = "ssh-public-key-callback"
)

// KeyboardInteractive implements AuthMethod by using a
// prompt/response sequence controlled by the server.
type KeyboardInteractive struct {
	User      string
	Challenge ssh.KeyboardInteractiveChallenge
	baseAuthMethod
}

func (a *KeyboardInteractive) Name() string {
	return KeyboardInteractiveName
}

func (a *KeyboardInteractive) String() string {
	return fmt.Sprintf("user: %s, name: %s", a.User, a.Name())
}

func (a *KeyboardInteractive) clientConfig() *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: a.User,
		Auth: []ssh.AuthMethod{ssh.KeyboardInteractiveChallenge(a.Challenge)},
	}
}

// Password implements AuthMethod by using the given password.
type Password struct {
	User string
	Pass string
	baseAuthMethod
}

func (a *Password) Name() string {
	return PasswordName
}

func (a *Password) String() string {
	return fmt.Sprintf("user: %s, name: %s", a.User, a.Name())
}

func (a *Password) clientConfig() *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: a.User,
		Auth: []ssh.AuthMethod{ssh.Password(a.Pass)},
	}
}

// PasswordCallback implements AuthMethod by using a callback
// to fetch the password.
type PasswordCallback struct {
	User     string
	Callback func() (pass string, err error)
	baseAuthMethod
}

func (a *PasswordCallback) Name() string {
	return PasswordCallbackName
}

func (a *PasswordCallback) String() string {
	return fmt.Sprintf("user: %s, name: %s", a.User, a.Name())
}

func (a *PasswordCallback) clientConfig() *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: a.User,
		Auth: []ssh.AuthMethod{ssh.PasswordCallback(a.Callback)},
	}
}

// PublicKeys implements AuthMethod by using the given key pairs.
type PublicKeys struct {
	User   string
	Signer ssh.Signer
	baseAuthMethod
}

// NewPublicKeys returns a PublicKeys from a PEM encoded private key. An
// encryption password should be given if the pemBytes contains a password
// encrypted PEM block otherwise password should be empty. It supports RSA
// (PKCS#1), DSA (OpenSSL), and ECDSA private keys.
func NewPublicKeys(user string, pemBytes []byte, password string) (AuthMethod, error) {
	block, _ := pem.Decode(pemBytes)
	if x509.IsEncryptedPEMBlock(block) {
		key, err := x509.DecryptPEMBlock(block, []byte(password))
		if err != nil {
			return nil, err
		}

		block = &pem.Block{Type: block.Type, Bytes: key}
		pemBytes = pem.EncodeToMemory(block)
	}

	signer, err := ssh.ParsePrivateKey(pemBytes)
	if err != nil {
		return nil, err
	}

	return &PublicKeys{User: user, Signer: signer}, nil
}

// NewPublicKeysFromFile returns a PublicKeys from a file containing a PEM
// encoded private key. An encryption password should be given if the pemBytes
// contains a password encrypted PEM block otherwise password should be empty.
func NewPublicKeysFromFile(user, pemFile, password string) (AuthMethod, error) {
	bytes, err := ioutil.ReadFile(pemFile)
	if err != nil {
		return nil, err
	}

	return NewPublicKeys(user, bytes, password)
}

func (a *PublicKeys) Name() string {
	return PublicKeysName
}

func (a *PublicKeys) String() string {
	return fmt.Sprintf("user: %s, name: %s", a.User, a.Name())
}

func (a *PublicKeys) clientConfig() *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: a.User,
		Auth: []ssh.AuthMethod{ssh.PublicKeys(a.Signer)},
	}
}

func username() (string, error) {
	var username string
	if user, err := user.Current(); err == nil {
		username = user.Username
	} else {
		username = os.Getenv("USER")
	}
	if username == "" {
		return "", errors.New("failed to get username")
	}
	return username, nil
}

// PublicKeysCallback implements AuthMethod by asking a
// ssh.agent.Agent to act as a signer.
type PublicKeysCallback struct {
	User     string
	Callback func() (signers []ssh.Signer, err error)
	baseAuthMethod
}

// NewSSHAgentAuth returns a PublicKeysCallback based on a SSH agent, it opens
// a pipe with the SSH agent and uses the pipe as the implementer of the public
// key callback function.
func NewSSHAgentAuth(u string) (AuthMethod, error) {
	var err error
	if u == "" {
		u, err = username()
		if err != nil {
			return nil, err
		}
	}

	a, _, err := sshagent.New()
	if err != nil {
		return nil, fmt.Errorf("error creating SSH agent: %q", err)
	}

	return &PublicKeysCallback{
		User:     u,
		Callback: a.Signers,
	}, nil
}

func (a *PublicKeysCallback) Name() string {
	return PublicKeysCallbackName
}

func (a *PublicKeysCallback) String() string {
	return fmt.Sprintf("user: %s, name: %s", a.User, a.Name())
}

func (a *PublicKeysCallback) clientConfig() *ssh.ClientConfig {
	return &ssh.ClientConfig{
		User: a.User,
		Auth: []ssh.AuthMethod{ssh.PublicKeysCallback(a.Callback)},
	}
}

// NewKnownHostsCallback returns ssh.HostKeyCallback based on a file based on a
// know_hosts file. http://man.openbsd.org/sshd#SSH_KNOWN_HOSTS_FILE_FORMAT
//
// If files is empty, the list of files will be read from the SSH_KNOWN_HOSTS
// environment variable, example:
//   /home/foo/custom_known_hosts_file:/etc/custom_known/hosts_file
//
// If SSH_KNOWN_HOSTS is not set the following file locations will be used:
//   ~/.ssh/known_hosts
//   /etc/ssh/ssh_known_hosts
func NewKnownHostsCallback(files ...string) (ssh.HostKeyCallback, error) {
	files, err := getDefaultKnownHostsFiles()
	if err != nil {
		return nil, err
	}

	files, err = filterKnownHostsFiles(files...)
	if err != nil {
		return nil, err
	}

	return knownhosts.New(files...)
}

func getDefaultKnownHostsFiles() ([]string, error) {
	files := filepath.SplitList(os.Getenv("SSH_KNOWN_HOSTS"))
	if len(files) != 0 {
		return files, nil
	}

	homeDirPath, err := homedir.Dir()
	if err != nil {
		return nil, err
	}

	return []string{
		filepath.Join(homeDirPath, "/.ssh/known_hosts"),
		"/etc/ssh/ssh_known_hosts",
	}, nil
}

func filterKnownHostsFiles(files ...string) ([]string, error) {
	var out []string
	for _, file := range files {
		_, err := os.Stat(file)
		if err == nil {
			out = append(out, file)
			continue
		}

		if !os.IsNotExist(err) {
			return nil, err
		}
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("unable to find any valid know_hosts file, set SSH_KNOWN_HOSTS env variable")
	}

	return out, nil
}

type baseAuthMethod struct {
	// HostKeyCallback is the function type used for verifying server keys.
	// If nil default callback will be create using NewKnownHostsHostKeyCallback
	// without argument.
	HostKeyCallback ssh.HostKeyCallback
}

func (m *baseAuthMethod) hostKeyCallback() (ssh.HostKeyCallback, error) {
	if m.HostKeyCallback == nil {
		return NewKnownHostsCallback()
	}

	return m.HostKeyCallback, nil
}
