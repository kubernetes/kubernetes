package catalog

const (
	QUESTION_TYPE = "question"
)

type Question struct {
	Resource

	Default string `json:"default,omitempty" yaml:"default,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Group string `json:"group,omitempty" yaml:"group,omitempty"`

	InvalidChars string `json:"invalidChars,omitempty" yaml:"invalid_chars,omitempty"`

	Label string `json:"label,omitempty" yaml:"label,omitempty"`

	Max int64 `json:"max,omitempty" yaml:"max,omitempty"`

	MaxLength int64 `json:"maxLength,omitempty" yaml:"max_length,omitempty"`

	Min int64 `json:"min,omitempty" yaml:"min,omitempty"`

	MinLength int64 `json:"minLength,omitempty" yaml:"min_length,omitempty"`

	Options []string `json:"options,omitempty" yaml:"options,omitempty"`

	Required bool `json:"required,omitempty" yaml:"required,omitempty"`

	Type string `json:"type,omitempty" yaml:"type,omitempty"`

	ValidChars string `json:"validChars,omitempty" yaml:"valid_chars,omitempty"`

	Variable string `json:"variable,omitempty" yaml:"variable,omitempty"`
}

type QuestionCollection struct {
	Collection
	Data []Question `json:"data,omitempty"`
}

type QuestionClient struct {
	rancherClient *RancherClient
}

type QuestionOperations interface {
	List(opts *ListOpts) (*QuestionCollection, error)
	Create(opts *Question) (*Question, error)
	Update(existing *Question, updates interface{}) (*Question, error)
	ById(id string) (*Question, error)
	Delete(container *Question) error
}

func newQuestionClient(rancherClient *RancherClient) *QuestionClient {
	return &QuestionClient{
		rancherClient: rancherClient,
	}
}

func (c *QuestionClient) Create(container *Question) (*Question, error) {
	resp := &Question{}
	err := c.rancherClient.doCreate(QUESTION_TYPE, container, resp)
	return resp, err
}

func (c *QuestionClient) Update(existing *Question, updates interface{}) (*Question, error) {
	resp := &Question{}
	err := c.rancherClient.doUpdate(QUESTION_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *QuestionClient) List(opts *ListOpts) (*QuestionCollection, error) {
	resp := &QuestionCollection{}
	err := c.rancherClient.doList(QUESTION_TYPE, opts, resp)
	return resp, err
}

func (c *QuestionClient) ById(id string) (*Question, error) {
	resp := &Question{}
	err := c.rancherClient.doById(QUESTION_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *QuestionClient) Delete(container *Question) error {
	return c.rancherClient.doResourceDelete(QUESTION_TYPE, &container.Resource)
}
