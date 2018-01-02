package mock

import (
	"net/http"
	"net/http/httptest"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/session"
)

// Session is a mock session which is used to hit the mock server
var Session = func() *session.Session {
	// server is the mock server that simply writes a 200 status back to the client
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	return session.Must(session.NewSession(&aws.Config{
		DisableSSL: aws.Bool(true),
		Endpoint:   aws.String(server.URL),
	}))
}()

// NewMockClient creates and initializes a client that will connect to the
// mock server
func NewMockClient(cfgs ...*aws.Config) *client.Client {
	c := Session.ClientConfig("Mock", cfgs...)

	svc := client.New(
		*c.Config,
		metadata.ClientInfo{
			ServiceName:   "Mock",
			SigningRegion: c.SigningRegion,
			Endpoint:      c.Endpoint,
			APIVersion:    "2015-12-08",
			JSONVersion:   "1.1",
			TargetPrefix:  "MockServer",
		},
		c.Handlers,
	)

	return svc
}
