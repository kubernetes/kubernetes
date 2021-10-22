package queues

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToQueueListQuery() (string, error)
}

// ListOpts params to be used with List
type ListOpts struct {
	// Limit instructs List to refrain from sending excessively large lists of queues
	Limit int `q:"limit,omitempty"`

	// Marker and Limit control paging. Marker instructs List where to start listing from.
	Marker string `q:"marker,omitempty"`

	// Specifies if showing the detailed information when querying queues
	Detailed bool `q:"detailed,omitempty"`
}

// ToQueueListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToQueueListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of queues.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToQueueListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	pager := pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return QueuePage{pagination.LinkedPageBase{PageResult: r}}

	})
	return pager
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToQueueCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies the queue creation parameters.
type CreateOpts struct {
	// The name of the queue to create.
	QueueName string `json:"queue_name" required:"true"`

	// The target incoming messages will be moved to when a message can’t
	// processed successfully after meet the max claim count is met.
	DeadLetterQueue string `json:"_dead_letter_queue,omitempty"`

	// The new TTL setting for messages when moved to dead letter queue.
	DeadLetterQueueMessagesTTL int `json:"_dead_letter_queue_messages_ttl,omitempty"`

	// The delay of messages defined for a queue. When the messages send to
	// the queue, it will be delayed for some times and means it can not be
	// claimed until the delay expired.
	DefaultMessageDelay int `json:"_default_message_delay,omitempty"`

	// The default TTL of messages defined for a queue, which will effect for
	// any messages posted to the queue.
	DefaultMessageTTL int `json:"_default_message_ttl" required:"true"`

	// The flavor name which can tell Zaqar which storage pool will be used
	// to create the queue.
	Flavor string `json:"_flavor,omitempty"`

	// The max number the message can be claimed.
	MaxClaimCount int `json:"_max_claim_count,omitempty"`

	// The max post size of messages defined for a queue, which will effect
	// for any messages posted to the queue.
	MaxMessagesPostSize int `json:"_max_messages_post_size,omitempty"`

	// Extra is free-form extra key/value pairs to describe the queue.
	Extra map[string]interface{} `json:"-"`
}

// ToQueueCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToQueueCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if opts.Extra != nil {
		for key, value := range opts.Extra {
			b[key] = value
		}

	}
	return b, nil
}

// Create requests the creation of a new queue.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToQueueCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	queueName := b["queue_name"].(string)
	delete(b, "queue_name")

	_, r.Err = client.Put(createURL(client, queueName), b, r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201, 204},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// update request.
type UpdateOptsBuilder interface {
	ToQueueUpdateMap() ([]map[string]interface{}, error)
}

// BatchUpdateOpts is an array of UpdateOpts.
type BatchUpdateOpts []UpdateOpts

// UpdateOpts is the struct responsible for updating a property of a queue.
type UpdateOpts struct {
	Op    UpdateOp    `json:"op" required:"true"`
	Path  string      `json:"path" required:"true"`
	Value interface{} `json:"value" required:"true"`
}

type UpdateOp string

const (
	ReplaceOp UpdateOp = "replace"
	AddOp     UpdateOp = "add"
	RemoveOp  UpdateOp = "remove"
)

// ToQueueUpdateMap constructs a request body from UpdateOpts.
func (opts BatchUpdateOpts) ToQueueUpdateMap() ([]map[string]interface{}, error) {
	queuesUpdates := make([]map[string]interface{}, len(opts))
	for i, queue := range opts {
		queueMap, err := queue.ToMap()
		if err != nil {
			return nil, err
		}
		queuesUpdates[i] = queueMap
	}
	return queuesUpdates, nil
}

// ToMap constructs a request body from UpdateOpts.
func (opts UpdateOpts) ToMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Update Updates the specified queue.
func Update(client *gophercloud.ServiceClient, queueName string, opts UpdateOptsBuilder) (r UpdateResult) {
	_, r.Err = client.Patch(updateURL(client, queueName), opts, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 204},
		MoreHeaders: map[string]string{
			"Content-Type": "application/openstack-messaging-v2.0-json-patch"},
	})
	return
}

// Get requests details on a single queue, by name.
func Get(client *gophercloud.ServiceClient, queueName string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, queueName), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete deletes the specified queue.
func Delete(client *gophercloud.ServiceClient, queueName string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, queueName), &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	return
}

// GetStats returns statistics for the specified queue.
func GetStats(client *gophercloud.ServiceClient, queueName string) (r StatResult) {
	_, r.Err = client.Get(statURL(client, queueName), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200}})
	return
}

type SharePath string

const (
	PathMessages      SharePath = "messages"
	PathClaims        SharePath = "claims"
	PathSubscriptions SharePath = "subscriptions"
)

type ShareMethod string

const (
	MethodGet   ShareMethod = "GET"
	MethodPatch ShareMethod = "PATCH"
	MethodPost  ShareMethod = "POST"
	MethodPut   ShareMethod = "PUT"
)

// ShareOpts specifies share creation parameters.
type ShareOpts struct {
	Paths   []SharePath   `json:"paths,omitempty"`
	Methods []ShareMethod `json:"methods,omitempty"`
	Expires string        `json:"expires,omitempty"`
}

// ShareOptsBuilder allows extensions to add additional attributes to the
// Share request.
type ShareOptsBuilder interface {
	ToQueueShareMap() (map[string]interface{}, error)
}

// ToShareQueueMap formats a ShareOpts structure into a request body.
func (opts ShareOpts) ToQueueShareMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Share creates a pre-signed URL for a given queue.
func Share(client *gophercloud.ServiceClient, queueName string, opts ShareOptsBuilder) (r ShareResult) {
	b, err := opts.ToQueueShareMap()
	if err != nil {
		r.Err = err
		return r
	}
	_, r.Err = client.Post(shareURL(client, queueName), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type PurgeResource string

const (
	ResourceMessages      PurgeResource = "messages"
	ResourceSubscriptions PurgeResource = "subscriptions"
)

// PurgeOpts specifies the purge parameters.
type PurgeOpts struct {
	ResourceTypes []PurgeResource `json:"resource_types" required:"true"`
}

// PurgeOptsBuilder allows extensions to add additional attributes to the
// Purge request.
type PurgeOptsBuilder interface {
	ToQueuePurgeMap() (map[string]interface{}, error)
}

// ToPurgeQueueMap formats a PurgeOpts structure into a request body
func (opts PurgeOpts) ToQueuePurgeMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return b, nil
}

// Purge purges particular resource of the queue.
func Purge(client *gophercloud.ServiceClient, queueName string, opts PurgeOptsBuilder) (r PurgeResult) {
	b, err := opts.ToQueuePurgeMap()
	if err != nil {
		r.Err = err
		return r
	}

	_, r.Err = client.Post(purgeURL(client, queueName), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	return
}
