package rdsutils

import (
	"fmt"
	"net/url"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
)

// ConnectionFormat is the type of connection that will be
// used to connect to the database
type ConnectionFormat string

// ConnectionFormat enums
const (
	NoConnectionFormat ConnectionFormat = ""
	TCPFormat          ConnectionFormat = "tcp"
)

// ErrNoConnectionFormat will be returned during build if no format had been
// specified
var ErrNoConnectionFormat = awserr.New("NoConnectionFormat", "No connection format was specified", nil)

// ConnectionStringBuilder is a builder that will construct a connection
// string with the provided parameters. params field is required to have
// a tls specification and allowCleartextPasswords must be set to true.
type ConnectionStringBuilder struct {
	dbName   string
	endpoint string
	region   string
	user     string
	creds    *credentials.Credentials

	connectFormat ConnectionFormat
	params        url.Values
}

// NewConnectionStringBuilder will return an ConnectionStringBuilder
func NewConnectionStringBuilder(endpoint, region, dbUser, dbName string, creds *credentials.Credentials) ConnectionStringBuilder {
	return ConnectionStringBuilder{
		dbName:   dbName,
		endpoint: endpoint,
		region:   region,
		user:     dbUser,
		creds:    creds,
	}
}

// WithEndpoint will return a builder with the given endpoint
func (b ConnectionStringBuilder) WithEndpoint(endpoint string) ConnectionStringBuilder {
	b.endpoint = endpoint
	return b
}

// WithRegion will return a builder with the given region
func (b ConnectionStringBuilder) WithRegion(region string) ConnectionStringBuilder {
	b.region = region
	return b
}

// WithUser will return a builder with the given user
func (b ConnectionStringBuilder) WithUser(user string) ConnectionStringBuilder {
	b.user = user
	return b
}

// WithDBName will return a builder with the given database name
func (b ConnectionStringBuilder) WithDBName(dbName string) ConnectionStringBuilder {
	b.dbName = dbName
	return b
}

// WithParams will return a builder with the given params. The parameters
// will be included in the connection query string
//
//	Example:
//	v := url.Values{}
//	v.Add("tls", "rds")
//	b := rdsutils.NewConnectionBuilder(endpoint, region, user, dbname, creds)
//	connectStr, err := b.WithParams(v).WithTCPFormat().Build()
func (b ConnectionStringBuilder) WithParams(params url.Values) ConnectionStringBuilder {
	b.params = params
	return b
}

// WithFormat will return a builder with the given connection format
func (b ConnectionStringBuilder) WithFormat(f ConnectionFormat) ConnectionStringBuilder {
	b.connectFormat = f
	return b
}

// WithTCPFormat will set the format to TCP and return the modified builder
func (b ConnectionStringBuilder) WithTCPFormat() ConnectionStringBuilder {
	return b.WithFormat(TCPFormat)
}

// Build will return a new connection string that can be used to open a connection
// to the desired database.
//
//	Example:
//	b := rdsutils.NewConnectionStringBuilder(endpoint, region, user, dbname, creds)
//	connectStr, err := b.WithTCPFormat().Build()
//	if err != nil {
//		panic(err)
//	}
//	const dbType = "mysql"
//	db, err := sql.Open(dbType, connectStr)
func (b ConnectionStringBuilder) Build() (string, error) {
	if b.connectFormat == NoConnectionFormat {
		return "", ErrNoConnectionFormat
	}

	authToken, err := BuildAuthToken(b.endpoint, b.region, b.user, b.creds)
	if err != nil {
		return "", err
	}

	connectionStr := fmt.Sprintf("%s:%s@%s(%s)/%s",
		b.user, authToken, string(b.connectFormat), b.endpoint, b.dbName,
	)

	if len(b.params) > 0 {
		connectionStr = fmt.Sprintf("%s?%s", connectionStr, b.params.Encode())
	}
	return connectionStr, nil
}
