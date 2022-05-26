package csm

import (
	"fmt"
	"strings"
	"sync"
)

var (
	lock sync.Mutex
)

const (
	// DefaultPort is used when no port is specified.
	DefaultPort = "31000"

	// DefaultHost is the host that will be used when none is specified.
	DefaultHost = "127.0.0.1"
)

// AddressWithDefaults returns a CSM address built from the host and port
// values. If the host or port is not set, default values will be used
// instead. If host is "localhost" it will be replaced with "127.0.0.1".
func AddressWithDefaults(host, port string) string {
	if len(host) == 0 || strings.EqualFold(host, "localhost") {
		host = DefaultHost
	}

	if len(port) == 0 {
		port = DefaultPort
	}

	// Only IP6 host can contain a colon
	if strings.Contains(host, ":") {
		return "[" + host + "]:" + port
	}

	return host + ":" + port
}

// Start will start a long running go routine to capture
// client side metrics. Calling start multiple time will only
// start the metric listener once and will panic if a different
// client ID or port is passed in.
//
//		r, err := csm.Start("clientID", "127.0.0.1:31000")
//		if err != nil {
//			panic(fmt.Errorf("expected no error, but received %v", err))
//		}
//		sess := session.NewSession()
//		r.InjectHandlers(sess.Handlers)
//
//		svc := s3.New(sess)
//		out, err := svc.GetObject(&s3.GetObjectInput{
//			Bucket: aws.String("bucket"),
//			Key: aws.String("key"),
//		})
func Start(clientID string, url string) (*Reporter, error) {
	lock.Lock()
	defer lock.Unlock()

	if sender == nil {
		sender = newReporter(clientID, url)
	} else {
		if sender.clientID != clientID {
			panic(fmt.Errorf("inconsistent client IDs. %q was expected, but received %q", sender.clientID, clientID))
		}

		if sender.url != url {
			panic(fmt.Errorf("inconsistent URLs. %q was expected, but received %q", sender.url, url))
		}
	}

	if err := connect(url); err != nil {
		sender = nil
		return nil, err
	}

	return sender, nil
}

// Get will return a reporter if one exists, if one does not exist, nil will
// be returned.
func Get() *Reporter {
	lock.Lock()
	defer lock.Unlock()

	return sender
}
