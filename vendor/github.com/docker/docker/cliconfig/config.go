package cliconfig

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/pkg/homedir"
	"github.com/docker/docker/pkg/system"
)

const (
	// ConfigFile is the name of config file
	ConfigFileName = "config.json"
	oldConfigfile  = ".dockercfg"

	// This constant is only used for really old config files when the
	// URL wasn't saved as part of the config file and it was just
	// assumed to be this value.
	defaultIndexserver = "https://index.docker.io/v1/"
)

var (
	configDir = os.Getenv("DOCKER_CONFIG")
)

func init() {
	if configDir == "" {
		configDir = filepath.Join(homedir.Get(), ".docker")
	}
}

// ConfigDir returns the directory the configuration file is stored in
func ConfigDir() string {
	return configDir
}

// SetConfigDir sets the directory the configuration file is stored in
func SetConfigDir(dir string) {
	configDir = dir
}

// AuthConfig contains authorization information for connecting to a Registry
type AuthConfig struct {
	Username      string `json:"username,omitempty"`
	Password      string `json:"password,omitempty"`
	Auth          string `json:"auth"`
	Email         string `json:"email"`
	ServerAddress string `json:"serveraddress,omitempty"`
}

// ConfigFile ~/.docker/config.json file info
type ConfigFile struct {
	AuthConfigs map[string]AuthConfig `json:"auths"`
	HTTPHeaders map[string]string     `json:"HttpHeaders,omitempty"`
	filename    string                // Note: not serialized - for internal use only
}

// NewConfigFile initilizes an empty configuration file for the given filename 'fn'
func NewConfigFile(fn string) *ConfigFile {
	return &ConfigFile{
		AuthConfigs: make(map[string]AuthConfig),
		HTTPHeaders: make(map[string]string),
		filename:    fn,
	}
}

// Load reads the configuration files in the given directory, and sets up
// the auth config information and return values.
// FIXME: use the internal golang config parser
func Load(configDir string) (*ConfigFile, error) {
	if configDir == "" {
		configDir = ConfigDir()
	}

	configFile := ConfigFile{
		AuthConfigs: make(map[string]AuthConfig),
		filename:    filepath.Join(configDir, ConfigFileName),
	}

	// Try happy path first - latest config file
	if _, err := os.Stat(configFile.filename); err == nil {
		file, err := os.Open(configFile.filename)
		if err != nil {
			return &configFile, err
		}
		defer file.Close()

		if err := json.NewDecoder(file).Decode(&configFile); err != nil {
			return &configFile, err
		}

		for addr, ac := range configFile.AuthConfigs {
			ac.Username, ac.Password, err = DecodeAuth(ac.Auth)
			if err != nil {
				return &configFile, err
			}
			ac.Auth = ""
			ac.ServerAddress = addr
			configFile.AuthConfigs[addr] = ac
		}

		return &configFile, nil
	} else if !os.IsNotExist(err) {
		// if file is there but we can't stat it for any reason other
		// than it doesn't exist then stop
		return &configFile, err
	}

	// Can't find latest config file so check for the old one
	confFile := filepath.Join(homedir.Get(), oldConfigfile)
	if _, err := os.Stat(confFile); err != nil {
		return &configFile, nil //missing file is not an error
	}

	b, err := ioutil.ReadFile(confFile)
	if err != nil {
		return &configFile, err
	}

	if err := json.Unmarshal(b, &configFile.AuthConfigs); err != nil {
		arr := strings.Split(string(b), "\n")
		if len(arr) < 2 {
			return &configFile, fmt.Errorf("The Auth config file is empty")
		}
		authConfig := AuthConfig{}
		origAuth := strings.Split(arr[0], " = ")
		if len(origAuth) != 2 {
			return &configFile, fmt.Errorf("Invalid Auth config file")
		}
		authConfig.Username, authConfig.Password, err = DecodeAuth(origAuth[1])
		if err != nil {
			return &configFile, err
		}
		origEmail := strings.Split(arr[1], " = ")
		if len(origEmail) != 2 {
			return &configFile, fmt.Errorf("Invalid Auth config file")
		}
		authConfig.Email = origEmail[1]
		authConfig.ServerAddress = defaultIndexserver
		configFile.AuthConfigs[defaultIndexserver] = authConfig
	} else {
		for k, authConfig := range configFile.AuthConfigs {
			authConfig.Username, authConfig.Password, err = DecodeAuth(authConfig.Auth)
			if err != nil {
				return &configFile, err
			}
			authConfig.Auth = ""
			authConfig.ServerAddress = k
			configFile.AuthConfigs[k] = authConfig
		}
	}
	return &configFile, nil
}

// Save encodes and writes out all the authorization information
func (configFile *ConfigFile) Save() error {
	// Encode sensitive data into a new/temp struct
	tmpAuthConfigs := make(map[string]AuthConfig, len(configFile.AuthConfigs))
	for k, authConfig := range configFile.AuthConfigs {
		authCopy := authConfig
		// encode and save the authstring, while blanking out the original fields
		authCopy.Auth = EncodeAuth(&authCopy)
		authCopy.Username = ""
		authCopy.Password = ""
		authCopy.ServerAddress = ""
		tmpAuthConfigs[k] = authCopy
	}

	saveAuthConfigs := configFile.AuthConfigs
	configFile.AuthConfigs = tmpAuthConfigs
	defer func() { configFile.AuthConfigs = saveAuthConfigs }()

	data, err := json.MarshalIndent(configFile, "", "\t")
	if err != nil {
		return err
	}

	if err := system.MkdirAll(filepath.Dir(configFile.filename), 0700); err != nil {
		return err
	}

	if err := ioutil.WriteFile(configFile.filename, data, 0600); err != nil {
		return err
	}

	return nil
}

// Filename returns the name of the configuration file
func (configFile *ConfigFile) Filename() string {
	return configFile.filename
}

// EncodeAuth creates a base64 encoded string to containing authorization information
func EncodeAuth(authConfig *AuthConfig) string {
	authStr := authConfig.Username + ":" + authConfig.Password
	msg := []byte(authStr)
	encoded := make([]byte, base64.StdEncoding.EncodedLen(len(msg)))
	base64.StdEncoding.Encode(encoded, msg)
	return string(encoded)
}

// DecodeAuth decodes a base64 encoded string and returns username and password
func DecodeAuth(authStr string) (string, string, error) {
	decLen := base64.StdEncoding.DecodedLen(len(authStr))
	decoded := make([]byte, decLen)
	authByte := []byte(authStr)
	n, err := base64.StdEncoding.Decode(decoded, authByte)
	if err != nil {
		return "", "", err
	}
	if n > decLen {
		return "", "", fmt.Errorf("Something went wrong decoding auth config")
	}
	arr := strings.SplitN(string(decoded), ":", 2)
	if len(arr) != 2 {
		return "", "", fmt.Errorf("Invalid auth configuration file")
	}
	password := strings.Trim(arr[1], "\x00")
	return arr[0], password, nil
}
