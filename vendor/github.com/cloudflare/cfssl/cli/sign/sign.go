// Package sign implements the sign command.
package sign

import (
	"encoding/json"
	"errors"
	"io/ioutil"

	"github.com/cloudflare/cfssl/certdb/dbconf"
	certsql "github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/universal"

	"github.com/jmoiron/sqlx"
)

// Usage text of 'cfssl sign'
var signerUsageText = `cfssl sign -- signs a client cert with a host name by a given CA and CA key

Usage of sign:
        cfssl sign -ca cert -ca-key key [mutual-tls-cert cert] [mutual-tls-key key] [-config config] [-profile profile] [-hostname hostname] [-db-config db-config] CSR [SUBJECT]
        cfssl sign -remote remote_host [mutual-tls-cert cert] [mutual-tls-key key] [-config config] [-profile profile] [-label label] [-hostname hostname] CSR [SUBJECT]

Arguments:
        CSR:        PEM file for certificate request, use '-' for reading PEM from stdin.

Note: CSR can also be supplied via flag values; flag value will take precedence over the argument.

SUBJECT is an optional file containing subject information to use for the certificate instead of the subject information in the CSR.

Flags:
`

// Flags of 'cfssl sign'
var signerFlags = []string{"hostname", "csr", "ca", "ca-key", "config", "profile", "label", "remote",
	"mutual-tls-cert", "mutual-tls-key", "db-config"}

// SignerFromConfigAndDB takes the Config and creates the appropriate
// signer.Signer object with a specified db
func SignerFromConfigAndDB(c cli.Config, db *sqlx.DB) (signer.Signer, error) {
	// If there is a config, use its signing policy. Otherwise create a default policy.
	var policy *config.Signing
	if c.CFG != nil {
		policy = c.CFG.Signing
	} else {
		policy = &config.Signing{
			Profiles: map[string]*config.SigningProfile{},
			Default:  config.DefaultConfig(),
		}
	}

	// Make sure the policy reflects the new remote
	if c.Remote != "" {
		err := policy.OverrideRemotes(c.Remote)
		if err != nil {
			log.Infof("Invalid remote %v, reverting to configuration default", c.Remote)
			return nil, err
		}
	}

	if c.MutualTLSCertFile != "" && c.MutualTLSKeyFile != "" {
		err := policy.SetClientCertKeyPairFromFile(c.MutualTLSCertFile, c.MutualTLSKeyFile)
		if err != nil {
			log.Infof("Invalid mutual-tls-cert: %s or mutual-tls-key: %s, defaulting to no client auth", c.MutualTLSCertFile, c.MutualTLSKeyFile)
			return nil, err
		}
		log.Infof("Using client auth with mutual-tls-cert: %s and mutual-tls-key: %s", c.MutualTLSCertFile, c.MutualTLSKeyFile)
	}

	if c.TLSRemoteCAs != "" {
		err := policy.SetRemoteCAsFromFile(c.TLSRemoteCAs)
		if err != nil {
			log.Infof("Invalid tls-remote-ca: %s, defaulting to system trust store", c.TLSRemoteCAs)
			return nil, err
		}
		log.Infof("Using trusted CA from tls-remote-ca: %s", c.TLSRemoteCAs)
	}

	s, err := universal.NewSigner(cli.RootFromConfig(&c), policy)
	if err != nil {
		return nil, err
	}

	if db != nil {
		dbAccessor := certsql.NewAccessor(db)
		s.SetDBAccessor(dbAccessor)
	}

	return s, nil
}

// SignerFromConfig takes the Config and creates the appropriate
// signer.Signer object
func SignerFromConfig(c cli.Config) (s signer.Signer, err error) {
	var db *sqlx.DB
	if c.DBConfigFile != "" {
		db, err = dbconf.DBFromConfig(c.DBConfigFile)
		if err != nil {
			return nil, err
		}
	}
	return SignerFromConfigAndDB(c, db)
}

// signerMain is the main CLI of signer functionality.
// [TODO: zi] Decide whether to drop the argument list and only use flags to specify all the inputs.
func signerMain(args []string, c cli.Config) (err error) {
	if c.CSRFile == "" {
		c.CSRFile, args, err = cli.PopFirstArgument(args)
		if err != nil {
			return
		}
	}

	var subjectData *signer.Subject
	if len(args) > 0 {
		var subjectFile string
		subjectFile, args, err = cli.PopFirstArgument(args)
		if err != nil {
			return
		}
		if len(args) > 0 {
			return errors.New("too many arguments are provided, please check with usage")
		}

		var subjectJSON []byte
		subjectJSON, err = ioutil.ReadFile(subjectFile)
		if err != nil {
			return
		}

		subjectData = new(signer.Subject)
		err = json.Unmarshal(subjectJSON, subjectData)
		if err != nil {
			return
		}
	}

	csr, err := cli.ReadStdin(c.CSRFile)
	if err != nil {
		return
	}

	// Remote can be forced on the command line or in the config
	if c.Remote == "" && c.CFG == nil {
		if c.CAFile == "" {
			log.Error("need CA certificate (provide one with -ca)")
			return
		}

		if c.CAKeyFile == "" {
			log.Error("need CA key (provide one with -ca-key)")
			return
		}
	}

	s, err := SignerFromConfig(c)
	if err != nil {
		return
	}

	req := signer.SignRequest{
		Hosts:   signer.SplitHosts(c.Hostname),
		Request: string(csr),
		Subject: subjectData,
		Profile: c.Profile,
		Label:   c.Label,
	}
	cert, err := s.Sign(req)
	if err != nil {
		return
	}
	cli.PrintCert(nil, csr, cert)
	return
}

// Command assembles the definition of Command 'sign'
var Command = &cli.Command{UsageText: signerUsageText, Flags: signerFlags, Main: signerMain}
