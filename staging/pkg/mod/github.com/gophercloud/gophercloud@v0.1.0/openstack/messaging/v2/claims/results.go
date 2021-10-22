package claims

import "github.com/gophercloud/gophercloud"

func (r CreateResult) Extract() ([]Messages, error) {
	var s struct {
		Messages []Messages `json:"messages"`
	}
	err := r.ExtractInto(&s)
	return s.Messages, err
}

func (r GetResult) Extract() (*Claim, error) {
	var s *Claim
	err := r.ExtractInto(&s)
	return s, err
}

// CreateResult is the response of a Create operations.
type CreateResult struct {
	gophercloud.Result
}

// GetResult is the response of a Get operations.
type GetResult struct {
	gophercloud.Result
}

// UpdateResult is the response of a Update operations.
type UpdateResult struct {
	gophercloud.ErrResult
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

type Messages struct {
	Age  float32                `json:"age"`
	Href string                 `json:"href"`
	TTL  int                    `json:"ttl"`
	Body map[string]interface{} `json:"body"`
}

type Claim struct {
	Age      float32    `json:"age"`
	Href     string     `json:"href"`
	Messages []Messages `json:"messages"`
	TTL      int        `json:"ttl"`
}
