// Package rdsutils is used to generate authentication tokens used to
// connect to a givent Amazon Relational Database Service (RDS) database.
//
// Before using the authentication please visit the docs here to ensure
// the database has the proper policies to allow for IAM token authentication.
// https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.html#UsingWithRDS.IAMDBAuth.Availability
//
// When building the connection string, there are two required parameters that are needed to be set on the query.
//	* tls
//	* allowCleartextPasswords must be set to true
//
//	Example creating a basic auth token with the builder:
//	v := url.Values{}
//	v.Add("tls", "tls_profile_name")
//	v.Add("allowCleartextPasswords", "true")
//	b := rdsutils.NewConnectionStringBuilder(endpoint, region, user, dbname, creds)
//	connectStr, err := b.WithTCPFormat().WithParams(v).Build()
package rdsutils
