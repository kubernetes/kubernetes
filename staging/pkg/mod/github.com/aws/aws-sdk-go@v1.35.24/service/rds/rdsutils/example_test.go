// +build example

package rdsutils_test

import (
	"database/sql"
	"flag"
	"fmt"
	"net/url"
	"os"

	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rds/rdsutils"
)

// ExampleConnectionStringBuilder contains usage of assuming a role and using
// that to build the auth token.
//
// For MySQL you many need to configure your MySQL driver with the TLS
// certificate to ensure verified TLS handshake,
// https://s3.amazonaws.com/rds-downloads/rds-combined-ca-bundle.pem
//
// Usage:
//	./main -user "iamuser" -dbname "foo" -region "us-west-2" -rolearn "arn" -endpoint "dbendpoint" -port 3306
func ExampleConnectionStringBuilder() {
	userPtr := flag.String("user", "", "user of the credentials")
	regionPtr := flag.String("region", "us-east-1", "region to be used when grabbing sts creds")
	roleArnPtr := flag.String("rolearn", "", "role arn to be used when grabbing sts creds")
	endpointPtr := flag.String("endpoint", "", "DB endpoint to be connected to")
	portPtr := flag.Int("port", 3306, "DB port to be connected to")
	tablePtr := flag.String("table", "test_table", "DB table to query against")
	dbNamePtr := flag.String("dbname", "", "DB name to query against")
	flag.Parse()

	// Check required flags. Will exit with status code 1 if
	// required field isn't set.
	if err := requiredFlags(
		userPtr,
		regionPtr,
		roleArnPtr,
		endpointPtr,
		portPtr,
		dbNamePtr,
	); err != nil {
		fmt.Printf("Error: %v\n\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}

	sess := session.Must(session.NewSession())
	creds := stscreds.NewCredentials(sess, *roleArnPtr)

	v := url.Values{}
	// required fields for DB connection
	v.Add("tls", "rds")
	v.Add("allowCleartextPasswords", "true")
	endpoint := fmt.Sprintf("%s:%d", *endpointPtr, *portPtr)

	b := rdsutils.NewConnectionStringBuilder(endpoint, *regionPtr, *userPtr, *dbNamePtr, creds)
	connectStr, err := b.WithTCPFormat().WithParams(v).Build()

	const dbType = "mysql"
	db, err := sql.Open(dbType, connectStr)
	// if an error is encountered here, then most likely security groups are incorrect
	// in the database.
	if err != nil {
		panic(fmt.Errorf("failed to open connection to the database"))
	}

	rows, err := db.Query(fmt.Sprintf("SELECT * FROM %s  LIMIT 1", *tablePtr))
	if err != nil {
		panic(fmt.Errorf("failed to select from table, %q, with %v", *tablePtr, err))
	}

	for rows.Next() {
		columns, err := rows.Columns()
		if err != nil {
			panic(fmt.Errorf("failed to read columns from row: %v", err))
		}

		fmt.Printf("rows colums:\n%d\n", len(columns))
	}
}

func requiredFlags(flags ...interface{}) error {
	for _, f := range flags {
		switch f.(type) {
		case nil:
			return fmt.Errorf("one or more required flags were not set")
		}
	}
	return nil
}
