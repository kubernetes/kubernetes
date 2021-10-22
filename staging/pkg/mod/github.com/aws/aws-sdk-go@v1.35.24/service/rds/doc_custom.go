// IAM User or Role Database Authentication
//
// The rdsutil package's BuildAuthToken function provides a connection
// authentication token builder. Given an endpoint of the RDS database,
// AWS region, DB user, and AWS credentials the function will create an
// presigned URL to use as the authentication token for the database's
// connection.
//
// The following example shows how to use BuildAuthToken to create an authentication
// token for connecting to a MySQL database in RDS.
//
//   authToken, err := rdsutils.BuildAuthToken(dbEndpoint, awsRegion, dbUser, awsCreds)
//
//   // Create the MySQL DNS string for the DB connection
//   // user:password@protocol(endpoint)/dbname?<params>
//   dnsStr = fmt.Sprintf("%s:%s@tcp(%s)/%s?tls=true",
//      dbUser, authToken, dbEndpoint, dbName,
//   )
//
//   // Use db to perform SQL operations on database
//   db, err := sql.Open("mysql", dnsStr)
//
// See rdsutil package for more information.
// http://docs.aws.amazon.com/sdk-for-go/api/service/rds/rdsutils/
package rds
