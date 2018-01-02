// Package config contains the multi-root configuration file parser.
package config

import (
	"bufio"
	"crypto"
	"crypto/x509"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/cloudflare/cfssl/certdb/dbconf"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/helpers/derhelpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/whitelist"

	"github.com/cloudflare/redoctober/client"
	"github.com/cloudflare/redoctober/core"
	"github.com/jmoiron/sqlx"
)

// RawMap is shorthand for the type used as a map from string to raw Root struct.
type RawMap map[string]map[string]string

var (
	configSection    = regexp.MustCompile("^\\s*\\[\\s*(\\w+)\\s*\\]\\s*$")
	quotedConfigLine = regexp.MustCompile("^\\s*(\\w+)\\s*=\\s*[\"'](.*)[\"']\\s*$")
	configLine       = regexp.MustCompile("^\\s*(\\w+)\\s*=\\s*(.*)\\s*$")
	commentLine      = regexp.MustCompile("^#.*$")
	blankLine        = regexp.MustCompile("^\\s*$")

	defaultSection = "default"
)

// ParseToRawMap takes the filename as a string and returns a RawMap.
func ParseToRawMap(fileName string) (cfg RawMap, err error) {
	var file *os.File

	cfg = make(RawMap, 0)
	file, err = os.Open(fileName)
	if err != nil {
		return
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	var currentSection string
	for scanner.Scan() {
		line := scanner.Text()

		if commentLine.MatchString(line) {
			continue
		} else if blankLine.MatchString(line) {
			continue
		} else if configSection.MatchString(line) {
			section := configSection.ReplaceAllString(line, "$1")
			if !cfg.SectionInConfig(section) {
				cfg[section] = make(map[string]string, 0)
			}
			currentSection = section
		} else if configLine.MatchString(line) {
			regex := configLine
			if quotedConfigLine.MatchString(line) {
				regex = quotedConfigLine
			}
			if currentSection == "" {
				currentSection = defaultSection
				if !cfg.SectionInConfig(currentSection) {
					cfg[currentSection] = make(map[string]string, 0)
				}
			}
			key := regex.ReplaceAllString(line, "$1")
			val := regex.ReplaceAllString(line, "$2")
			cfg[currentSection][key] = val
		} else {
			err = fmt.Errorf("invalid config file")
			break
		}
	}
	return
}

// SectionInConfig determines whether a section is in the configuration.
func (c *RawMap) SectionInConfig(section string) bool {
	for s := range *c {
		if section == s {
			return true
		}
	}
	return false
}

// A Root represents a single certificate authority root key pair.
type Root struct {
	PrivateKey  crypto.Signer
	Certificate *x509.Certificate
	Config      *config.Signing
	ACL         whitelist.NetACL
	DB          *sqlx.DB
}

// LoadRoot parses a config structure into a Root structure
func LoadRoot(cfg map[string]string) (*Root, error) {
	var root Root
	var err error
	spec, ok := cfg["private"]
	if !ok {
		return nil, ErrMissingPrivateKey
	}

	certPath, ok := cfg["certificate"]
	if !ok {
		return nil, ErrMissingCertificatePath
	}

	configPath, ok := cfg["config"]
	if !ok {
		return nil, ErrMissingConfigPath
	}

	root.PrivateKey, err = parsePrivateKeySpec(spec, cfg)
	if err != nil {
		return nil, err
	}

	in, err := ioutil.ReadFile(certPath)
	if err != nil {
		return nil, err
	}

	root.Certificate, err = helpers.ParseCertificatePEM(in)
	if err != nil {
		return nil, err
	}

	conf, err := config.LoadFile(configPath)
	if err != nil {
		return nil, err
	}
	root.Config = conf.Signing

	nets := cfg["nets"]
	if nets != "" {
		root.ACL, err = parseACL(nets)
		if err != nil {
			return nil, err
		}
	}

	dbConfig := cfg["dbconfig"]
	if dbConfig != "" {
		db, err := dbconf.DBFromConfig(dbConfig)
		if err != nil {
			return nil, err
		}
		root.DB = db
	}

	return &root, nil
}

func parsePrivateKeySpec(spec string, cfg map[string]string) (crypto.Signer, error) {
	specURL, err := url.Parse(spec)
	if err != nil {
		return nil, err
	}

	var priv crypto.Signer
	switch specURL.Scheme {
	case "file":
		// A file spec will be parsed such that the root
		// directory of a relative path will be stored as the
		// hostname, and the remainder of the file's path is
		// stored in the Path field.
		log.Debug("loading private key file", specURL.Path)
		path := filepath.Join(specURL.Host, specURL.Path)
		in, err := ioutil.ReadFile(path)
		if err != nil {
			return nil, err
		}

		log.Debug("attempting to load PEM-encoded private key")
		priv, err = helpers.ParsePrivateKeyPEM(in)
		if err != nil {
			log.Debug("file is not a PEM-encoded private key")
			log.Debug("attempting to load DER-encoded private key")
			priv, err = derhelpers.ParsePrivateKeyDER(in)
			if err != nil {
				return nil, err
			}
		}
		log.Debug("loaded private key")
		return priv, nil
	case "rofile":
		log.Warning("Red October support is currently experimental")
		path := filepath.Join(specURL.Host, specURL.Path)
		in, err := ioutil.ReadFile(path)
		if err != nil {
			return nil, err
		}

		roServer := cfg["ro_server"]
		if roServer == "" {
			return nil, errors.New("config: no RedOctober server available")
		}

		// roCAPath can be empty; if it is, the client uses
		// the system default CA roots.
		roCAPath := cfg["ro_ca"]

		roUser := cfg["ro_user"]
		if roUser == "" {
			return nil, errors.New("config: no RedOctober user available")
		}

		roPass := cfg["ro_pass"]
		if roPass == "" {
			return nil, errors.New("config: no RedOctober passphrase available")
		}

		log.Debug("decrypting key via RedOctober Server")
		roClient, err := client.NewRemoteServer(roServer, roCAPath)
		if err != nil {
			return nil, err
		}

		req := core.DecryptRequest{
			Name:     roUser,
			Password: roPass,
			Data:     in,
		}
		in, err = roClient.DecryptIntoData(req)
		if err != nil {
			return nil, err
		}

		return priv, nil
	default:
		return nil, ErrUnsupportedScheme
	}
}

func parseACL(nets string) (whitelist.NetACL, error) {
	wl := whitelist.NewBasicNet()
	netList := strings.Split(nets, ",")
	for i := range netList {
		netList[i] = strings.TrimSpace(netList[i])
		_, n, err := net.ParseCIDR(netList[i])
		if err != nil {
			return nil, err
		}

		wl.Add(n)
	}

	return wl, nil
}

// A RootList associates a set of labels with the appropriate private
// keys and their certificates.
type RootList map[string]*Root

var (
	// ErrMissingPrivateKey indicates that the configuration is
	// missing a private key specifier.
	ErrMissingPrivateKey = errors.New("config: root is missing private key spec")

	// ErrMissingCertificatePath indicates that the configuration
	// is missing a certificate specifier.
	ErrMissingCertificatePath = errors.New("config: root is missing certificate path")

	// ErrMissingConfigPath indicates that the configuration lacks
	// a valid CFSSL configuration.
	ErrMissingConfigPath = errors.New("config: root is missing configuration file path")

	// ErrInvalidConfig indicates the configuration is invalid.
	ErrInvalidConfig = errors.New("config: invalid configuration")

	// ErrUnsupportedScheme indicates a private key scheme that is not currently supported.
	ErrUnsupportedScheme = errors.New("config: unsupported private key scheme")
)

// Parse loads a RootList from a file.
func Parse(filename string) (RootList, error) {
	cfgMap, err := ParseToRawMap(filename)
	if err != nil {
		return nil, err
	}

	var rootList = RootList{}
	for label, entries := range cfgMap {
		root, err := LoadRoot(entries)
		if err != nil {
			return nil, err
		}

		rootList[label] = root
	}

	return rootList, nil
}
