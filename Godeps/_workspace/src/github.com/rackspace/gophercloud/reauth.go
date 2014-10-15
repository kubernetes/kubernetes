package gophercloud

import (
	"github.com/racker/perigee"
)

// WithReauth wraps a Perigee request fragment with logic to perform re-authentication
// if it's deemed necessary.
//
// Do not confuse this function with WithReauth()!  Although they work together to support reauthentication,
// WithReauth() actually contains the decision-making logic to determine when to perform a reauth,
// while WithReauthHandler() is used to configure what a reauth actually entails.
func (c *Context) WithReauth(ap AccessProvider, f func() error) error {
	err := f()
	cause, ok := err.(*perigee.UnexpectedResponseCodeError)
	if ok && cause.Actual == 401 {
		err = c.reauthHandler(ap)
		if err == nil {
			err = f()
		}
	}
	return err
}

// This is like WithReauth above but returns a perigee Response object
func (c *Context) ResponseWithReauth(ap AccessProvider, f func() (*perigee.Response, error)) (*perigee.Response, error) {
	response, err := f()
	cause, ok := err.(*perigee.UnexpectedResponseCodeError)
	if ok && cause.Actual == 401 {
		err = c.reauthHandler(ap)
		if err == nil {
			response, err = f()
		}
	}
	return response, err
}
