package oauth

import (
	"errors"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/klog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"

	oauthv1 "github.com/openshift/api/oauth/v1"
	userv1 "github.com/openshift/api/user/v1"
	oauthclient "github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1"
	oauthclientlister "github.com/openshift/client-go/oauth/listers/oauth/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/authentication/oauth/rankedset"
)

var errTimedout = errors.New("token timed out")

// Implements rankedset.Item
var _ = rankedset.Item(&tokenData{})

type tokenData struct {
	token *oauthv1.OAuthAccessToken
	seen  time.Time
}

func (a *tokenData) timeout() time.Time {
	return a.token.CreationTimestamp.Time.Add(time.Duration(a.token.InactivityTimeoutSeconds) * time.Second)
}

func (a *tokenData) Key() string {
	return a.token.Name
}

func (a *tokenData) Rank() int64 {
	return a.timeout().Unix()
}

func timeoutAsDuration(timeout int32) time.Duration {
	return time.Duration(timeout) * time.Second
}

type TimeoutValidator struct {
	oauthClients   oauthclientlister.OAuthClientLister
	tokens         oauthclient.OAuthAccessTokenInterface
	tokenChannel   chan *tokenData
	data           *rankedset.RankedSet
	defaultTimeout time.Duration
	tickerInterval time.Duration

	// fields that are used to have a deterministic order of events in unit tests
	flushHandler    func(flushHorizon time.Time) // allows us to decorate this func during unit tests
	putTokenHandler func(td *tokenData)          // allows us to decorate this func during unit tests
	clock           clock.Clock                  // allows us to control time during unit tests
}

func NewTimeoutValidator(tokens oauthclient.OAuthAccessTokenInterface, oauthClients oauthclientlister.OAuthClientLister, defaultTimeout int32, minValidTimeout int32) *TimeoutValidator {
	a := &TimeoutValidator{
		oauthClients:   oauthClients,
		tokens:         tokens,
		tokenChannel:   make(chan *tokenData),
		data:           rankedset.New(),
		defaultTimeout: timeoutAsDuration(defaultTimeout),
		tickerInterval: timeoutAsDuration(minValidTimeout / 3), // we tick at least 3 times within each timeout period
		clock:          clock.RealClock{},
	}
	a.flushHandler = a.flush
	a.putTokenHandler = a.putToken
	klog.V(5).Infof("Token Timeout Validator primed with defaultTimeout=%s tickerInterval=%s", a.defaultTimeout, a.tickerInterval)
	return a
}

// Validate is called with a token when it is seen by an authenticator
// it touches only the tokenChannel so it is safe to call from other threads
func (a *TimeoutValidator) Validate(token *oauthv1.OAuthAccessToken, _ *userv1.User) error {
	if token.InactivityTimeoutSeconds == 0 {
		// We care only if the token was created with a timeout to start with
		return nil
	}

	td := &tokenData{
		token: token,
		seen:  a.clock.Now(),
	}
	if td.timeout().Before(td.seen) {
		return errTimedout
	}

	if token.ExpiresIn != 0 && token.ExpiresIn <= int64(token.InactivityTimeoutSeconds) {
		// skip if the timeout is already larger than expiration deadline
		return nil
	}
	// After a positive timeout check we need to update the timeout and
	// schedule an update so that we can either set or update the Timeout
	// we do that launching a micro goroutine to avoid blocking
	go a.putTokenHandler(td)

	return nil
}

func (a *TimeoutValidator) putToken(td *tokenData) {
	a.tokenChannel <- td
}

func (a *TimeoutValidator) clientTimeout(name string) time.Duration {
	oauthClient, err := a.oauthClients.Get(name)
	if err != nil {
		klog.V(5).Infof("Failed to fetch OAuthClient %q for timeout value: %v", name, err)
		return a.defaultTimeout
	}
	if oauthClient.AccessTokenInactivityTimeoutSeconds == nil {
		return a.defaultTimeout
	}
	return timeoutAsDuration(*oauthClient.AccessTokenInactivityTimeoutSeconds)
}

func (a *TimeoutValidator) update(td *tokenData) error {
	// Obtain the timeout interval for this client
	delta := a.clientTimeout(td.token.ClientName)
	// if delta is 0 it means the OAuthClient has been changed to the
	// no-timeout value. In this case we set newTimeout also to 0 so
	// that the token will no longer timeout once updated.
	newTimeout := int32(0)
	if delta > 0 {
		// InactivityTimeoutSeconds is the number of seconds since creation:
		// InactivityTimeoutSeconds = Seen(Time) - CreationTimestamp(Time) + delta(Duration)
		newTimeout = int32((td.seen.Sub(td.token.CreationTimestamp.Time) + delta) / time.Second)
	}
	// We need to get the token again here because it may have changed in the
	// DB and we need to verify it is still worth updating
	token, err := a.tokens.Get(td.token.Name, v1.GetOptions{})
	if err != nil {
		return err
	}
	if newTimeout != 0 && token.InactivityTimeoutSeconds >= newTimeout {
		// if the token was already updated with a higher or equal timeout we
		// do not have anything to do
		return nil
	}
	token.InactivityTimeoutSeconds = newTimeout
	_, err = a.tokens.Update(token)
	return err
}

func (a *TimeoutValidator) flush(flushHorizon time.Time) {
	// flush all tokens that are about to expire before the flushHorizon.
	// Typically the flushHorizon is set to a time slightly past the next
	// ticker interval, so that not token ends up timing out between flushes
	klog.V(5).Infof("Flushing tokens timing out before %s", flushHorizon)

	// grab all tokens that need to be update in this flush interval
	// and remove them from the stored data, they either flush now or never
	tokenList := a.data.LessThan(flushHorizon.Unix(), true)

	var retryList []*tokenData
	flushedTokens := 0

	for _, item := range tokenList {
		td := item.(*tokenData)
		err := a.update(td)
		// not logging the full errors here as it would leak the token.
		switch {
		case err == nil:
			flushedTokens++
		case apierrors.IsConflict(err) || apierrors.IsServerTimeout(err):
			klog.V(5).Infof("Token update deferred for token belonging to %s",
				td.token.UserName)
			retryList = append(retryList, td)
		default:
			klog.V(5).Infof("Token timeout for user=%q client=%q scopes=%v was not updated",
				td.token.UserName, td.token.ClientName, td.token.Scopes)
		}
	}

	// we try once more and if it still fails we stop trying here and defer
	// to a future regular update if the token is used again
	for _, td := range retryList {
		err := a.update(td)
		if err != nil {
			klog.V(5).Infof("Token timeout for user=%q client=%q scopes=%v was not updated",
				td.token.UserName, td.token.ClientName, td.token.Scopes)
		} else {
			flushedTokens++
		}
	}

	klog.V(5).Infof("Successfully flushed %d tokens out of %d",
		flushedTokens, len(tokenList))
}

func (a *TimeoutValidator) nextTick() time.Time {
	// Add a small safety Margin so flushes tend to
	// overlap a little rather than have gaps
	return a.clock.Now().Add(a.tickerInterval + 10*time.Second)
}

func (a *TimeoutValidator) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	klog.V(5).Infof("Started Token Timeout Flush Handling thread!")

	ticker := a.clock.NewTicker(a.tickerInterval)
	// make sure to kill the ticker when we exit
	defer ticker.Stop()

	nextTick := a.nextTick()

	for {
		select {
		case <-stopCh:
			// if channel closes terminate
			return

		case td := <-a.tokenChannel:
			a.data.Insert(td)
			// if this token is going to time out before the timer, flush now
			tokenTimeout := td.timeout()
			if tokenTimeout.Before(nextTick) {
				klog.V(5).Infof("Timeout for user=%q client=%q scopes=%v falls before next ticker (%s < %s), forcing flush!",
					td.token.UserName, td.token.ClientName, td.token.Scopes, tokenTimeout, nextTick)
				a.flushHandler(nextTick)
			}

		case <-ticker.C():
			nextTick = a.nextTick()
			a.flushHandler(nextTick)
		}
	}
}
