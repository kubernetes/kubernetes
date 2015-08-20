package cli

import (
	"flag"
	"os"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/signer/pkcs11"
	"github.com/cloudflare/cfssl/signer/universal"
)

// Config is a type to hold flag values used by cfssl commands.
type Config struct {
	Hostname          string
	CertFile          string
	CSRFile           string
	CAFile            string
	CAKeyFile         string
	KeyFile           string
	IntermediatesFile string
	CABundleFile      string
	IntBundleFile     string
	Address           string
	Port              int
	Password          string
	ConfigFile        string
	CFG               *config.Config
	Profile           string
	IsCA              bool
	IntDir            string
	Flavor            string
	Metadata          string
	Domain            string
	IP                string
	Remote            string
	Label             string
	AuthKey           string
	Module            string
	Token             string
	PIN               string
	PKCS11Label       string
	ResponderFile     string
	Status            string
	Reason            int
	RevokedAt         string
	Interval          int64
	List              bool
	Family            string
	Timeout           time.Duration
	Scanner           string
	Responses         string
	Path              string
	UseLocal          bool
	Usage             string
}

// registerFlags defines all cfssl command flags and associates their values with variables.
func registerFlags(c *Config, f *flag.FlagSet) {
	f.StringVar(&c.Hostname, "hostname", "", "Hostname for the cert, could be a comma-separated hostname list")
	f.StringVar(&c.CertFile, "cert", "", "Client certificate that contains the public key")
	f.StringVar(&c.CSRFile, "csr", "", "Certificate signature request file for new public key")
	f.StringVar(&c.CAFile, "ca", "ca.pem", "CA used to sign the new certificate")
	f.StringVar(&c.CAKeyFile, "ca-key", "ca-key.pem", "CA private key")
	f.StringVar(&c.KeyFile, "key", "", "private key for the certificate")
	f.StringVar(&c.IntermediatesFile, "intermediates", "", "intermediate certs")
	f.StringVar(&c.CABundleFile, "ca-bundle", "/etc/cfssl/ca-bundle.crt", "Bundle to be used for root certificates pool")
	f.StringVar(&c.IntBundleFile, "int-bundle", "/etc/cfssl/int-bundle.crt", "Bundle to be used for intermediate certificates pool")
	f.StringVar(&c.Address, "address", "127.0.0.1", "Address to bind")
	f.IntVar(&c.Port, "port", 8888, "Port to bind")
	f.StringVar(&c.ConfigFile, "config", "", "path to configuration file")
	f.StringVar(&c.Profile, "profile", "", "signing profile to use")
	f.BoolVar(&c.IsCA, "initca", false, "initialise new CA")
	f.StringVar(&c.IntDir, "int-dir", "/etc/cfssl/intermediates", "specify intermediates directory")
	f.StringVar(&c.Flavor, "flavor", "ubiquitous", "Bundle Flavor: ubiquitous, optimal and force.")
	f.StringVar(&c.Metadata, "metadata", "/etc/cfssl/ca-bundle.crt.metadata", "Metadata file for root certificate presence. The content of the file is a json dictionary (k,v): each key k is SHA-1 digest of a root certificate while value v is a list of key store filenames.")
	f.StringVar(&c.Domain, "domain", "", "remote server domain name")
	f.StringVar(&c.IP, "ip", "", "remote server ip")
	f.StringVar(&c.Remote, "remote", "", "remote CFSSL server")
	f.StringVar(&c.Label, "label", "", "key label to use in remote CFSSL server")
	f.StringVar(&c.AuthKey, "authkey", "", "key to authenticate requests to remote CFSSL server")
	f.StringVar(&c.ResponderFile, "responder", "", "Certificate for OCSP responder")
	f.StringVar(&c.Status, "status", "good", "Status of the certificate: good, revoked, unknown")
	f.IntVar(&c.Reason, "reason", 0, "Reason code for revocation")
	f.StringVar(&c.RevokedAt, "revoked-at", "now", "Date of revocation (YYYY-MM-DD)")
	f.Int64Var(&c.Interval, "interval", int64(4*helpers.OneDay), "Interval between OCSP updates, in seconds (default: 4 days)")
	f.BoolVar(&c.List, "list", false, "list possible scanners")
	f.StringVar(&c.Family, "family", "", "scanner family regular expression")
	f.StringVar(&c.Scanner, "scanner", "", "scanner regular expression")
	f.DurationVar(&c.Timeout, "timeout", 0, "duration (ns, us, ms, s, m, h) to scan each host before timing out")
	f.StringVar(&c.Responses, "responses", "", "file to load OCSP responses from")
	f.StringVar(&c.Path, "path", "/", "Path on which the server will listen")
	f.StringVar(&c.Password, "password", "0", "Password for accessing PKCS #12 data passed to bundler")
	f.BoolVar(&c.UseLocal, "uselocal", false, "serve local static files as opposed to compiled ones")
	f.StringVar(&c.Usage, "usage", "dev", "usage of private key")

	if pkcs11.Enabled {
		f.StringVar(&c.Module, "pkcs11-module", "", "PKCS #11 module")
		f.StringVar(&c.Token, "pkcs11-token", "", "PKCS #11 token")
		f.StringVar(&c.PIN, "pkcs11-pin", os.Getenv("USER_PIN"), "PKCS #11 user PIN")
		f.StringVar(&c.PKCS11Label, "pkcs11-label", "", "PKCS #11 label")
	}
}

// RootFromConfig returns a universal signer Root structure that can
// be used to produce a signer.
func RootFromConfig(c *Config) universal.Root {
	return universal.Root{
		Config: map[string]string{
			"pkcs11-module":   c.Module,
			"pkcs11-token":    c.Token,
			"pkcs11-label":    c.PKCS11Label,
			"pkcs11-user-pin": c.PIN,
			"cert-file":       c.CAFile,
			"key-file":        c.CAKeyFile,
		},
		ForceRemote: c.Remote != "",
	}
}
