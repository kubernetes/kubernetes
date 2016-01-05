package crammd5

import (
	"crypto/hmac"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"io"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/auth/callback"
	"github.com/mesos/mesos-go/auth/sasl/mech"
)

var (
	Name = "CRAM-MD5" // name this mechanism is registered with

	//TODO(jdef) is this a generic SASL error? if so, move it up to mech
	challengeDataRequired = errors.New("challenge data may not be empty")
)

func init() {
	mech.Register(Name, newInstance)
}

type mechanism struct {
	handler callback.Handler
}

func (m *mechanism) Handler() callback.Handler {
	return m.handler
}

func (m *mechanism) Discard() {
	// noop
}

func newInstance(h callback.Handler) (mech.Interface, mech.StepFunc, error) {
	m := &mechanism{
		handler: h,
	}
	fn := func(m mech.Interface, data []byte) (mech.StepFunc, []byte, error) {
		// noop: no initialization needed
		return challengeResponse, nil, nil
	}
	return m, fn, nil
}

// algorithm lifted from wikipedia: http://en.wikipedia.org/wiki/CRAM-MD5
// except that the SASL mechanism used by Mesos doesn't leverage base64 encoding
func challengeResponse(m mech.Interface, data []byte) (mech.StepFunc, []byte, error) {
	if len(data) == 0 {
		return mech.IllegalState, nil, challengeDataRequired
	}
	decoded := string(data)
	log.V(4).Infof("challenge(decoded): %s", decoded) // for deep debugging only

	username := callback.NewName()
	secret := callback.NewPassword()

	if err := m.Handler().Handle(username, secret); err != nil {
		return mech.IllegalState, nil, err
	}
	hash := hmac.New(md5.New, secret.Get())
	if _, err := io.WriteString(hash, decoded); err != nil {
		return mech.IllegalState, nil, err
	}

	codes := hex.EncodeToString(hash.Sum(nil))
	msg := username.Get() + " " + codes
	return nil, []byte(msg), nil
}
