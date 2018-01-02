// +build example,skip

package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"

	"github.com/go-sql-driver/mysql"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rds/rdsutils"
)

// Usage ./iam_authentication <region> <db user> <db name> <endpoint to database> <iam arn>
func main() {
	if len(os.Args) < 5 {
		log.Println("USAGE ERROR: go run concatenateObjects.go <region> <endpoint to database> <iam arn>")
		os.Exit(1)
	}

	awsRegion := os.Args[1]
	dbUser := os.Args[2]
	dbName := os.Args[3]
	dbEndpoint := os.Args[4]
	awsCreds := stscreds.NewCredentials(session.New(&aws.Config{Region: &awsRegion}), os.Args[5])
	authToken, err := rdsutils.BuildAuthToken(dbEndpoint, awsRegion, dbUser, awsCreds)

	// Create the MySQL DNS string for the DB connection
	// user:password@protocol(endpoint)/dbname?<params>
	dnsStr := fmt.Sprintf("%s:%s@tcp(%s)/%s?tls=true",
		dbUser, authToken, dbEndpoint, dbName,
	)

	driver := mysql.MySQLDriver{}
	_ = driver
	// Use db to perform SQL operations on database
	if _, err = sql.Open("mysql", dnsStr); err != nil {
		panic(err)
	}

	fmt.Println("Successfully opened connection to database")
}
