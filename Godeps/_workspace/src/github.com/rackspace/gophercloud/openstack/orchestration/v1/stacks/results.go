package stacks

import (
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// CreatedStack represents the object extracted from a Create operation.
type CreatedStack struct {
	ID    string             `mapstructure:"id"`
	Links []gophercloud.Link `mapstructure:"links"`
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a CreatedStack object and is called after a
// Create operation.
func (r CreateResult) Extract() (*CreatedStack, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Stack *CreatedStack `mapstructure:"stack"`
	}

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return res.Stack, nil
}

// AdoptResult represents the result of an Adopt operation. AdoptResult has the
// same form as CreateResult.
type AdoptResult struct {
	CreateResult
}

// StackPage is a pagination.Pager that is returned from a call to the List function.
type StackPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Stacks.
func (r StackPage) IsEmpty() (bool, error) {
	stacks, err := ExtractStacks(r)
	if err != nil {
		return true, err
	}
	return len(stacks) == 0, nil
}

// ListedStack represents an element in the slice extracted from a List operation.
type ListedStack struct {
	CreationTime time.Time          `mapstructure:"-"`
	Description  string             `mapstructure:"description"`
	ID           string             `mapstructure:"id"`
	Links        []gophercloud.Link `mapstructure:"links"`
	Name         string             `mapstructure:"stack_name"`
	Status       string             `mapstructure:"stack_status"`
	StatusReason string             `mapstructure:"stack_status_reason"`
	UpdatedTime  time.Time          `mapstructure:"-"`
}

// ExtractStacks extracts and returns a slice of ListedStack. It is used while iterating
// over a stacks.List call.
func ExtractStacks(page pagination.Page) ([]ListedStack, error) {
	casted := page.(StackPage).Body

	var res struct {
		Stacks []ListedStack `mapstructure:"stacks"`
	}

	err := mapstructure.Decode(page.(StackPage).Body, &res)
	if err != nil {
		return nil, err
	}

	var rawStacks []interface{}
	switch casted.(type) {
	case map[string]interface{}:
		rawStacks = casted.(map[string]interface{})["stacks"].([]interface{})
	case map[string][]interface{}:
		rawStacks = casted.(map[string][]interface{})["stacks"]
	default:
		return res.Stacks, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawStacks {
		thisStack := (rawStacks[i]).(map[string]interface{})

		if t, ok := thisStack["creation_time"].(string); ok && t != "" {
			creationTime, err := time.Parse(gophercloud.STACK_TIME_FMT, t)
			if err != nil {
				return res.Stacks, err
			}
			res.Stacks[i].CreationTime = creationTime
		}

		if t, ok := thisStack["updated_time"].(string); ok && t != "" {
			updatedTime, err := time.Parse(gophercloud.STACK_TIME_FMT, t)
			if err != nil {
				return res.Stacks, err
			}
			res.Stacks[i].UpdatedTime = updatedTime
		}
	}

	return res.Stacks, nil
}

// RetrievedStack represents the object extracted from a Get operation.
type RetrievedStack struct {
	Capabilities        []interface{}            `mapstructure:"capabilities"`
	CreationTime        time.Time                `mapstructure:"-"`
	Description         string                   `mapstructure:"description"`
	DisableRollback     bool                     `mapstructure:"disable_rollback"`
	ID                  string                   `mapstructure:"id"`
	Links               []gophercloud.Link       `mapstructure:"links"`
	NotificationTopics  []interface{}            `mapstructure:"notification_topics"`
	Outputs             []map[string]interface{} `mapstructure:"outputs"`
	Parameters          map[string]string        `mapstructure:"parameters"`
	Name                string                   `mapstructure:"stack_name"`
	Status              string                   `mapstructure:"stack_status"`
	StatusReason        string                   `mapstructure:"stack_status_reason"`
	TemplateDescription string                   `mapstructure:"template_description"`
	Timeout             int                      `mapstructure:"timeout_mins"`
	UpdatedTime         time.Time                `mapstructure:"-"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a RetrievedStack object and is called after a
// Get operation.
func (r GetResult) Extract() (*RetrievedStack, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Stack *RetrievedStack `mapstructure:"stack"`
	}

	config := &mapstructure.DecoderConfig{
		Result:           &res,
		WeaklyTypedInput: true,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return nil, err
	}

	if err := decoder.Decode(r.Body); err != nil {
		return nil, err
	}

	b := r.Body.(map[string]interface{})["stack"].(map[string]interface{})

	if date, ok := b["creation_time"]; ok && date != nil {
		t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
		if err != nil {
			return nil, err
		}
		res.Stack.CreationTime = t
	}

	if date, ok := b["updated_time"]; ok && date != nil {
		t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
		if err != nil {
			return nil, err
		}
		res.Stack.UpdatedTime = t
	}

	return res.Stack, err
}

// UpdateResult represents the result of a Update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// PreviewedStack represents the result of a Preview operation.
type PreviewedStack struct {
	Capabilities        []interface{}            `mapstructure:"capabilities"`
	CreationTime        time.Time                `mapstructure:"-"`
	Description         string                   `mapstructure:"description"`
	DisableRollback     bool                     `mapstructure:"disable_rollback"`
	ID                  string                   `mapstructure:"id"`
	Links               []gophercloud.Link       `mapstructure:"links"`
	Name                string                   `mapstructure:"stack_name"`
	NotificationTopics  []interface{}            `mapstructure:"notification_topics"`
	Parameters          map[string]string        `mapstructure:"parameters"`
	Resources           []map[string]interface{} `mapstructure:"resources"`
	Status              string                   `mapstructure:"stack_status"`
	StatusReason        string                   `mapstructure:"stack_status_reason"`
	TemplateDescription string                   `mapstructure:"template_description"`
	Timeout             int                      `mapstructure:"timeout_mins"`
	UpdatedTime         time.Time                `mapstructure:"-"`
}

// PreviewResult represents the result of a Preview operation.
type PreviewResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a PreviewedStack object and is called after a
// Preview operation.
func (r PreviewResult) Extract() (*PreviewedStack, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Stack *PreviewedStack `mapstructure:"stack"`
	}

	config := &mapstructure.DecoderConfig{
		Result:           &res,
		WeaklyTypedInput: true,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return nil, err
	}

	if err := decoder.Decode(r.Body); err != nil {
		return nil, err
	}

	b := r.Body.(map[string]interface{})["stack"].(map[string]interface{})

	if date, ok := b["creation_time"]; ok && date != nil {
		t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
		if err != nil {
			return nil, err
		}
		res.Stack.CreationTime = t
	}

	if date, ok := b["updated_time"]; ok && date != nil {
		t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
		if err != nil {
			return nil, err
		}
		res.Stack.UpdatedTime = t
	}

	return res.Stack, err
}

// AbandonedStack represents the result of an Abandon operation.
type AbandonedStack struct {
	Status    string                 `mapstructure:"status"`
	Name      string                 `mapstructure:"name"`
	Template  map[string]interface{} `mapstructure:"template"`
	Action    string                 `mapstructure:"action"`
	ID        string                 `mapstructure:"id"`
	Resources map[string]interface{} `mapstructure:"resources"`
}

// AbandonResult represents the result of an Abandon operation.
type AbandonResult struct {
	gophercloud.Result
}

// Extract returns a pointer to an AbandonedStack object and is called after an
// Abandon operation.
func (r AbandonResult) Extract() (*AbandonedStack, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res AbandonedStack

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}

// String converts an AbandonResult to a string. This is useful to when passing
// the result of an Abandon operation to an AdoptOpts AdoptStackData field.
func (r AbandonResult) String() (string, error) {
	out, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	return string(out), nil
}
