package api

import (
	"errors"
	"math/rand"
	"sync"
	"time"
)

var (
	ErrRenewerMissingInput  = errors.New("missing input to renewer")
	ErrRenewerMissingSecret = errors.New("missing secret to renew")
	ErrRenewerNotRenewable  = errors.New("secret is not renewable")
	ErrRenewerNoSecretData  = errors.New("returned empty secret data")

	// DefaultRenewerGrace is the default grace period
	DefaultRenewerGrace = 15 * time.Second

	// DefaultRenewerRenewBuffer is the default size of the buffer for renew
	// messages on the channel.
	DefaultRenewerRenewBuffer = 5
)

// Renewer is a process for renewing a secret.
//
// 	renewer, err := client.NewRenewer(&RenewerInput{
// 		Secret: mySecret,
// 	})
// 	go renewer.Renew()
// 	defer renewer.Stop()
//
// 	for {
// 		select {
// 		case err := <-renewer.DoneCh():
// 			if err != nil {
// 				log.Fatal(err)
// 			}
//
// 			// Renewal is now over
// 		case renewal := <-renewer.RenewCh():
// 			log.Printf("Successfully renewed: %#v", renewal)
// 		}
// 	}
//
//
// The `DoneCh` will return if renewal fails or if the remaining lease duration
// after a renewal is less than or equal to the grace (in number of seconds). In
// both cases, the caller should attempt a re-read of the secret. Clients should
// check the return value of the channel to see if renewal was successful.
type Renewer struct {
	l sync.Mutex

	client  *Client
	secret  *Secret
	grace   time.Duration
	random  *rand.Rand
	doneCh  chan error
	renewCh chan *RenewOutput

	stopped bool
	stopCh  chan struct{}
}

// RenewerInput is used as input to the renew function.
type RenewerInput struct {
	// Secret is the secret to renew
	Secret *Secret

	// Grace is a minimum renewal before returning so the upstream client
	// can do a re-read. This can be used to prevent clients from waiting
	// too long to read a new credential and incur downtime.
	Grace time.Duration

	// Rand is the randomizer to use for underlying randomization. If not
	// provided, one will be generated and seeded automatically. If provided, it
	// is assumed to have already been seeded.
	Rand *rand.Rand

	// RenewBuffer is the size of the buffered channel where renew messages are
	// dispatched.
	RenewBuffer int
}

// RenewOutput is the metadata returned to the client (if it's listening) to
// renew messages.
type RenewOutput struct {
	// RenewedAt is the timestamp when the renewal took place (UTC).
	RenewedAt time.Time

	// Secret is the underlying renewal data. It's the same struct as all data
	// that is returned from Vault, but since this is renewal data, it will not
	// usually include the secret itself.
	Secret *Secret
}

// NewRenewer creates a new renewer from the given input.
func (c *Client) NewRenewer(i *RenewerInput) (*Renewer, error) {
	if i == nil {
		return nil, ErrRenewerMissingInput
	}

	secret := i.Secret
	if secret == nil {
		return nil, ErrRenewerMissingSecret
	}

	grace := i.Grace
	if grace == 0 {
		grace = DefaultRenewerGrace
	}

	random := i.Rand
	if random == nil {
		random = rand.New(rand.NewSource(int64(time.Now().Nanosecond())))
	}

	renewBuffer := i.RenewBuffer
	if renewBuffer == 0 {
		renewBuffer = DefaultRenewerRenewBuffer
	}

	return &Renewer{
		client:  c,
		secret:  secret,
		grace:   grace,
		random:  random,
		doneCh:  make(chan error, 1),
		renewCh: make(chan *RenewOutput, renewBuffer),

		stopped: false,
		stopCh:  make(chan struct{}),
	}, nil
}

// DoneCh returns the channel where the renewer will publish when renewal stops.
// If there is an error, this will be an error.
func (r *Renewer) DoneCh() <-chan error {
	return r.doneCh
}

// RenewCh is a channel that receives a message when a successful renewal takes
// place and includes metadata about the renewal.
func (r *Renewer) RenewCh() <-chan *RenewOutput {
	return r.renewCh
}

// Stop stops the renewer.
func (r *Renewer) Stop() {
	r.l.Lock()
	if !r.stopped {
		close(r.stopCh)
		r.stopped = true
	}
	r.l.Unlock()
}

// Renew starts a background process for renewing this secret. When the secret
// is has auth data, this attempts to renew the auth (token). When the secret
// has a lease, this attempts to renew the lease.
func (r *Renewer) Renew() {
	var result error
	if r.secret.Auth != nil {
		result = r.renewAuth()
	} else {
		result = r.renewLease()
	}

	select {
	case r.doneCh <- result:
	case <-r.stopCh:
	}
}

// renewAuth is a helper for renewing authentication.
func (r *Renewer) renewAuth() error {
	if !r.secret.Auth.Renewable || r.secret.Auth.ClientToken == "" {
		return ErrRenewerNotRenewable
	}

	client, token := r.client, r.secret.Auth.ClientToken

	for {
		// Check if we are stopped.
		select {
		case <-r.stopCh:
			return nil
		default:
		}

		// Renew the auth.
		renewal, err := client.Auth().Token().RenewTokenAsSelf(token, 0)
		if err != nil {
			return err
		}

		// Push a message that a renewal took place.
		select {
		case r.renewCh <- &RenewOutput{time.Now().UTC(), renewal}:
		default:
		}

		// Somehow, sometimes, this happens.
		if renewal == nil || renewal.Auth == nil {
			return ErrRenewerNoSecretData
		}

		// Do nothing if we are not renewable
		if !renewal.Auth.Renewable {
			return ErrRenewerNotRenewable
		}

		// Grab the lease duration and sleep duration - note that we grab the auth
		// lease duration, not the secret lease duration.
		leaseDuration := time.Duration(renewal.Auth.LeaseDuration) * time.Second
		sleepDuration := r.sleepDuration(leaseDuration)

		// If we are within grace, return now.
		if leaseDuration <= r.grace || sleepDuration <= r.grace {
			return nil
		}

		select {
		case <-r.stopCh:
			return nil
		case <-time.After(sleepDuration):
			continue
		}
	}
}

// renewLease is a helper for renewing a lease.
func (r *Renewer) renewLease() error {
	if !r.secret.Renewable || r.secret.LeaseID == "" {
		return ErrRenewerNotRenewable
	}

	client, leaseID := r.client, r.secret.LeaseID

	for {
		// Check if we are stopped.
		select {
		case <-r.stopCh:
			return nil
		default:
		}

		// Renew the lease.
		renewal, err := client.Sys().Renew(leaseID, 0)
		if err != nil {
			return err
		}

		// Push a message that a renewal took place.
		select {
		case r.renewCh <- &RenewOutput{time.Now().UTC(), renewal}:
		default:
		}

		// Somehow, sometimes, this happens.
		if renewal == nil {
			return ErrRenewerNoSecretData
		}

		// Do nothing if we are not renewable
		if !renewal.Renewable {
			return ErrRenewerNotRenewable
		}

		// Grab the lease duration and sleep duration
		leaseDuration := time.Duration(renewal.LeaseDuration) * time.Second
		sleepDuration := r.sleepDuration(leaseDuration)

		// If we are within grace, return now.
		if leaseDuration <= r.grace || sleepDuration <= r.grace {
			return nil
		}

		select {
		case <-r.stopCh:
			return nil
		case <-time.After(sleepDuration):
			continue
		}
	}
}

// sleepDuration calculates the time to sleep given the base lease duration. The
// base is the resulting lease duration. It will be reduced to 1/3 and
// multiplied by a random float between 0.0 and 1.0. This extra randomness
// prevents multiple clients from all trying to renew simultaneously.
func (r *Renewer) sleepDuration(base time.Duration) time.Duration {
	sleep := float64(base)

	// Renew at 1/3 the remaining lease. This will give us an opportunity to retry
	// at least one more time should the first renewal fail.
	sleep = sleep / 3.0

	// Use a randomness so many clients do not hit Vault simultaneously.
	sleep = sleep * (r.random.Float64() + 1) / 2.0

	return time.Duration(sleep)
}
