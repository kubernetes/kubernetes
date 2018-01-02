package transport

import (
	"crypto/tls"
	"net"
	"time"

	"github.com/cloudflare/cfssl/log"
)

// A Listener is a TCP network listener for TLS-secured connections.
type Listener struct {
	*Transport
	net.Listener
}

// Listen sets up a new server. If an error is returned, it means
// the server isn't ready to begin listening.
func Listen(address string, tr *Transport) (*Listener, error) {
	var err error
	l := &Listener{Transport: tr}
	config, err := tr.getConfig()
	if err != nil {
		return nil, err
	}

	l.Listener, err = tls.Listen("tcp", address, config)
	return l, err
}

func (tr *Transport) getConfig() (*tls.Config, error) {
	if tr.ClientTrustStore != nil {
		log.Info("using client auth")
		return tr.TLSClientAuthServerConfig()
	}
	log.Info("not using client auth")
	return tr.TLSServerConfig()
}

// PollInterval is how often to check whether a new certificate has
// been found.
var PollInterval = 30 * time.Second

func pollWait(target time.Time) {
	for {
		<-time.After(PollInterval)
		if time.Now().After(target) {
			break
		}
	}
}

// AutoUpdate will automatically update the listener. If a non-nil
// certUpdates chan is provided, it will receive timestamps for
// reissued certificates. If errChan is non-nil, any errors that occur
// in the updater will be passed along.
func (l *Listener) AutoUpdate(certUpdates chan<- time.Time, errChan chan<- error) {
	defer func() {
		if r := recover(); r != nil {
			log.Criticalf("AutoUpdate panicked: %v", r)
		}
	}()

	for {
		// Wait until it's time to update the certificate.
		target := time.Now().Add(l.Lifespan())
		if PollInterval == 0 {
			<-time.After(l.Lifespan())
		} else {
			pollWait(target)
		}

		// Keep trying to update the certificate until it's
		// ready.
		for {
			log.Debug("refreshing certificate")
			err := l.RefreshKeys()
			if err == nil {
				break
			}

			delay := l.Transport.Backoff.Duration()
			log.Debugf("failed to update certificate, will try again in %s", delay)
			if errChan != nil {
				errChan <- err
			}

			<-time.After(delay)
		}

		if certUpdates != nil {
			certUpdates <- time.Now()
		}

		config, err := l.getConfig()
		if err != nil {
			log.Debug("immediately after getting a new certificate, the Transport is reporting errors: %v", err)
			if errChan != nil {
				errChan <- err
			}
		}

		address := l.Listener.Addr().String()
		lnet := l.Listener.Addr().Network()
		l.Listener, err = tls.Listen(lnet, address, config)
		if err != nil {
			log.Debug("immediately after getting a new certificate, the Transport is reporting errors: %v", err)
			if errChan != nil {
				errChan <- err
			}
		}

		log.Debug("listener: auto update of certificate complete")
		l.Transport.Backoff.Reset()
	}
}
