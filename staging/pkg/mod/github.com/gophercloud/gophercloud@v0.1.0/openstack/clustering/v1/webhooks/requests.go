package webhooks

import (
	"fmt"
	"net/url"

	"github.com/gophercloud/gophercloud"
)

// TriggerOpts represents options used for triggering an action
type TriggerOpts struct {
	V      string `q:"V" required:"true"`
	Params map[string]string
}

// TriggerOptsBuilder Query string builder interface for webhooks
type TriggerOptsBuilder interface {
	ToWebhookTriggerQuery() (string, error)
}

// Query string builder for webhooks
func (opts TriggerOpts) ToWebhookTriggerQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	params := q.Query()

	for k, v := range opts.Params {
		params.Add(k, v)
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), err
}

// Trigger an action represented by a webhook.
func Trigger(client *gophercloud.ServiceClient, id string, opts TriggerOptsBuilder) (r TriggerResult) {
	url := triggerURL(client, id)
	if opts != nil {
		query, err := opts.ToWebhookTriggerQuery()
		if err != nil {
			r.Err = err
			return
		}
		url += query
	} else {
		r.Err = fmt.Errorf("Must contain V for TriggerOpt")
		return
	}

	_, r.Err = client.Post(url, nil, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}
