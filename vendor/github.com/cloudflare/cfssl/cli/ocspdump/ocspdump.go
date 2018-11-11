// Package ocspdump implements the ocspdump command.
package ocspdump

import (
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/cloudflare/cfssl/certdb/dbconf"
	"github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/cli"
)

// Usage text of 'cfssl ocspdump'
var ocspdumpUsageText = `cfssl ocspdump -- generates a series of concatenated OCSP responses
for use with ocspserve from all OCSP responses in the cert db

Usage of ocspdump:
        cfssl ocspdump -db-config db-config

Flags:
`

// Flags of 'cfssl ocspdump'
var ocspdumpFlags = []string{"db-config"}

// ocspdumpMain is the main CLI of OCSP dump functionality.
func ocspdumpMain(args []string, c cli.Config) error {
	if c.DBConfigFile == "" {
		return errors.New("need DB config file (provide with -db-config)")
	}

	db, err := dbconf.DBFromConfig(c.DBConfigFile)
	if err != nil {
		return err
	}

	dbAccessor := sql.NewAccessor(db)
	records, err := dbAccessor.GetUnexpiredOCSPs()
	if err != nil {
		return err
	}
	for _, certRecord := range records {
		fmt.Printf("%s\n", base64.StdEncoding.EncodeToString([]byte(certRecord.Body)))
	}
	return nil
}

// Command assembles the definition of Command 'ocspdump'
var Command = &cli.Command{UsageText: ocspdumpUsageText, Flags: ocspdumpFlags, Main: ocspdumpMain}
