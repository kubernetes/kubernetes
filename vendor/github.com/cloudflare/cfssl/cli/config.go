package cli

import (
	"flag"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer/universal"
)

// Config is a type to hold flag values used by cfssl commands.
type Config struct {
	Hostname          string
	CertFile          string
	CSRFile           string
	CAFile            string
	CAKeyFile         string
	TLSCertFile       string
	TLSKeyFile        string
	MutualTLSCAFile   string
	MutualTLSCNRegex  string
	TLSRemoteCAs      string
	MutualTLSCertFile string
	MutualTLSKeyFile  string
	KeyFile           string
	IntermediatesFile string
	CABundleFile      string
	IntBundleFile     string
	Address           string
	Port              int
	MinTLSVersion     string
	Password          string
	ConfigFile        string
	CFG               *config.Config
	Profile           string
	IsCA              bool
	RenewCA           bool
	IntDir            string
	Flavor            string
	Metadata          string
	Domain            string
	IP                string
	Remote            string
	Label             string
	AuthKey           string
	ResponderFile     string
	ResponderKeyFile  string
	Status            string
	Reason            string
	RevokedAt         string
	Interval          time.Duration
	List              bool
	Family            string
	Timeout           time.Duration
	Scanner           string
	CSVFile           string
	NumWorkers        int
	MaxHosts          int
	Responses         string
	Path              string
	CRL               string
	Usage             string
	PGPPrivate        string
	PGPName           string
	Serial            string
	CNOverride        string
	AKI               string
	DBConfigFile      string
	CRLExpiration     time.Duration
	Disable     	  string
}

// registerFlags defines all cfssl command flags and associates their values with variables.
func registerFlags(c *Config, f *flag.FlagSet) {
	f.StringVar(&c.Hostname, "hostname", "", "Hostname for the cert, could be a comma-separated hostname list")
	f.StringVar(&c.CertFile, "cert", "", "Client certificate that contains the public key")
	f.StringVar(&c.CSRFile, "csr", "", "Certificate signature request file for new public key")
	f.StringVar(&c.CAFile, "ca", "", "CA used to sign the new certificate -- accepts '[file:]fname' or 'env:varname'")
	f.StringVar(&c.CAKeyFile, "ca-key", "", "CA private key -- accepts '[file:]fname' or 'env:varname'")
	f.StringVar(&c.TLSCertFile, "tls-cert", "", "Other endpoint CA to set up TLS protocol")
	f.StringVar(&c.TLSKeyFile, "tls-key", "", "Other endpoint CA private key")
	f.StringVar(&c.MutualTLSCAFile, "mutual-tls-ca", "", "Mutual TLS - require clients be signed by this CA ")
	f.StringVar(&c.MutualTLSCNRegex, "mutual-tls-cn", "", "Mutual TLS - regex for whitelist of allowed client CNs")
	f.StringVar(&c.TLSRemoteCAs, "tls-remote-ca", "", "CAs to trust for remote TLS requests")
	f.StringVar(&c.MutualTLSCertFile, "mutual-tls-client-cert", "", "Mutual TLS - client certificate to call remote instance requiring client certs")
	f.StringVar(&c.MutualTLSKeyFile, "mutual-tls-client-key", "", "Mutual TLS - client key to call remote instance requiring client certs")
	f.StringVar(&c.KeyFile, "key", "", "private key for the certificate")
	f.StringVar(&c.IntermediatesFile, "intermediates", "", "intermediate certs")
	f.StringVar(&c.CABundleFile, "ca-bundle", "", "path to root certificate store")
	f.StringVar(&c.IntBundleFile, "int-bundle", "", "path to intermediate certificate store")
	f.StringVar(&c.Address, "address", "127.0.0.1", "Address to bind")
	f.IntVar(&c.Port, "port", 8888, "Port to bind")
	f.StringVar(&c.MinTLSVersion, "min-tls-version", "", "Minimum version of TLS to use, defaults to 1.0")
	f.StringVar(&c.ConfigFile, "config", "", "path to configuration file")
	f.StringVar(&c.Profile, "profile", "", "signing profile to use")
	f.BoolVar(&c.IsCA, "initca", false, "initialise new CA")
	f.BoolVar(&c.RenewCA, "renewca", false, "re-generate a CA certificate from existing CA certificate/key")
	f.StringVar(&c.IntDir, "int-dir", "", "specify intermediates directory")
	f.StringVar(&c.Flavor, "flavor", "ubiquitous", "Bundle Flavor: ubiquitous, optimal and force.")
	f.StringVar(&c.Metadata, "metadata", "", "Metadata file for root certificate presence. The content of the file is a json dictionary (k,v): each key k is SHA-1 digest of a root certificate while value v is a list of key store filenames.")
	f.StringVar(&c.Domain, "domain", "", "remote server domain name")
	f.StringVar(&c.IP, "ip", "", "remote server ip")
	f.StringVar(&c.Remote, "remote", "", "remote CFSSL server")
	f.StringVar(&c.Label, "label", "", "key label to use in remote CFSSL server")
	f.StringVar(&c.AuthKey, "authkey", "", "key to authenticate requests to remote CFSSL server")
	f.StringVar(&c.ResponderFile, "responder", "", "Certificate for OCSP responder")
	f.StringVar(&c.ResponderKeyFile, "responder-key", "", "private key for OCSP responder certificate")
	f.StringVar(&c.Status, "status", "good", "Status of the certificate: good, revoked, unknown")
	f.StringVar(&c.Reason, "reason", "0", "Reason code for revocation")
	f.StringVar(&c.RevokedAt, "revoked-at", "now", "Date of revocation (YYYY-MM-DD)")
	f.DurationVar(&c.Interval, "interval", 4*helpers.OneDay, "Interval between OCSP updates (default: 96h)")
	f.BoolVar(&c.List, "list", false, "list possible scanners")
	f.StringVar(&c.Family, "family", "", "scanner family regular expression")
	f.StringVar(&c.Scanner, "scanner", "", "scanner regular expression")
	f.DurationVar(&c.Timeout, "timeout", 5*time.Minute, "duration (ns, us, ms, s, m, h) to scan each host before timing out")
	f.StringVar(&c.CSVFile, "csv", "", "file containing CSV of hosts")
	f.IntVar(&c.NumWorkers, "num-workers", 10, "number of workers to use for scan")
	f.IntVar(&c.MaxHosts, "max-hosts", 100, "maximum number of hosts to scan")
	f.StringVar(&c.Responses, "responses", "", "file to load OCSP responses from")
	f.StringVar(&c.Path, "path", "/", "Path on which the server will listen")
	f.StringVar(&c.CRL, "crl", "", "CRL URL Override")
	f.StringVar(&c.Password, "password", "0", "Password for accessing PKCS #12 data passed to bundler")
	f.StringVar(&c.Usage, "usage", "", "usage of private key")
	f.StringVar(&c.PGPPrivate, "pgp-private", "", "file to load a PGP Private key decryption")
	f.StringVar(&c.PGPName, "pgp-name", "", "PGP public key name, can be a comma-sepearted  key name list")
	f.StringVar(&c.Serial, "serial", "", "certificate serial number")
	f.StringVar(&c.CNOverride, "cn", "", "certificate common name (CN)")
	f.StringVar(&c.AKI, "aki", "", "certificate issuer (authority) key identifier")
	f.StringVar(&c.DBConfigFile, "db-config", "", "certificate db configuration file")
	f.DurationVar(&c.CRLExpiration, "expiry", 7*helpers.OneDay, "time from now after which the CRL will expire (default: one week)")
	f.IntVar(&log.Level, "loglevel", log.LevelInfo, "Log level (0 = DEBUG, 5 = FATAL)")
	f.StringVar(&c.Disable, "disable", "", "endpoints to disable")
}

// RootFromConfig returns a universal signer Root structure that can
// be used to produce a signer.
func RootFromConfig(c *Config) universal.Root {
	return universal.Root{
		Config: map[string]string{
			"cert-file": c.CAFile,
			"key-file":  c.CAKeyFile,
		},
		ForceRemote: c.Remote != "",
	}
}
