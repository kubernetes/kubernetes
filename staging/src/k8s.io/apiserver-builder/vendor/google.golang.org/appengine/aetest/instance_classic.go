// +build appengine

package aetest

import "appengine/aetest"

// NewInstance launches a running instance of api_server.py which can be used
// for multiple test Contexts that delegate all App Engine API calls to that
// instance.
// If opts is nil the default values are used.
func NewInstance(opts *Options) (Instance, error) {
	aetest.PrepareDevAppserver = PrepareDevAppserver
	var aeOpts *aetest.Options
	if opts != nil {
		aeOpts = &aetest.Options{
			AppID: opts.AppID,
			StronglyConsistentDatastore: opts.StronglyConsistentDatastore,
		}
	}
	return aetest.NewInstance(aeOpts)
}
