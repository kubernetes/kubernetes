package client

const (
	LDAPCONFIG_TYPE = "ldapconfig"
)

type Ldapconfig struct {
	Resource

	AccessMode string `json:"accessMode,omitempty" yaml:"access_mode,omitempty"`

	AllowedIdentities []interface{} `json:"allowedIdentities,omitempty" yaml:"allowed_identities,omitempty"`

	ConnectionTimeout int64 `json:"connectionTimeout,omitempty" yaml:"connection_timeout,omitempty"`

	Domain string `json:"domain,omitempty" yaml:"domain,omitempty"`

	Enabled bool `json:"enabled,omitempty" yaml:"enabled,omitempty"`

	GroupMemberMappingAttribute string `json:"groupMemberMappingAttribute,omitempty" yaml:"group_member_mapping_attribute,omitempty"`

	GroupNameField string `json:"groupNameField,omitempty" yaml:"group_name_field,omitempty"`

	GroupObjectClass string `json:"groupObjectClass,omitempty" yaml:"group_object_class,omitempty"`

	GroupSearchField string `json:"groupSearchField,omitempty" yaml:"group_search_field,omitempty"`

	LoginDomain string `json:"loginDomain,omitempty" yaml:"login_domain,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Port int64 `json:"port,omitempty" yaml:"port,omitempty"`

	Server string `json:"server,omitempty" yaml:"server,omitempty"`

	ServiceAccountPassword string `json:"serviceAccountPassword,omitempty" yaml:"service_account_password,omitempty"`

	ServiceAccountUsername string `json:"serviceAccountUsername,omitempty" yaml:"service_account_username,omitempty"`

	Tls bool `json:"tls,omitempty" yaml:"tls,omitempty"`

	UserDisabledBitMask int64 `json:"userDisabledBitMask,omitempty" yaml:"user_disabled_bit_mask,omitempty"`

	UserEnabledAttribute string `json:"userEnabledAttribute,omitempty" yaml:"user_enabled_attribute,omitempty"`

	UserLoginField string `json:"userLoginField,omitempty" yaml:"user_login_field,omitempty"`

	UserMemberAttribute string `json:"userMemberAttribute,omitempty" yaml:"user_member_attribute,omitempty"`

	UserNameField string `json:"userNameField,omitempty" yaml:"user_name_field,omitempty"`

	UserObjectClass string `json:"userObjectClass,omitempty" yaml:"user_object_class,omitempty"`

	UserSearchField string `json:"userSearchField,omitempty" yaml:"user_search_field,omitempty"`
}

type LdapconfigCollection struct {
	Collection
	Data []Ldapconfig `json:"data,omitempty"`
}

type LdapconfigClient struct {
	rancherClient *RancherClient
}

type LdapconfigOperations interface {
	List(opts *ListOpts) (*LdapconfigCollection, error)
	Create(opts *Ldapconfig) (*Ldapconfig, error)
	Update(existing *Ldapconfig, updates interface{}) (*Ldapconfig, error)
	ById(id string) (*Ldapconfig, error)
	Delete(container *Ldapconfig) error
}

func newLdapconfigClient(rancherClient *RancherClient) *LdapconfigClient {
	return &LdapconfigClient{
		rancherClient: rancherClient,
	}
}

func (c *LdapconfigClient) Create(container *Ldapconfig) (*Ldapconfig, error) {
	resp := &Ldapconfig{}
	err := c.rancherClient.doCreate(LDAPCONFIG_TYPE, container, resp)
	return resp, err
}

func (c *LdapconfigClient) Update(existing *Ldapconfig, updates interface{}) (*Ldapconfig, error) {
	resp := &Ldapconfig{}
	err := c.rancherClient.doUpdate(LDAPCONFIG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LdapconfigClient) List(opts *ListOpts) (*LdapconfigCollection, error) {
	resp := &LdapconfigCollection{}
	err := c.rancherClient.doList(LDAPCONFIG_TYPE, opts, resp)
	return resp, err
}

func (c *LdapconfigClient) ById(id string) (*Ldapconfig, error) {
	resp := &Ldapconfig{}
	err := c.rancherClient.doById(LDAPCONFIG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LdapconfigClient) Delete(container *Ldapconfig) error {
	return c.rancherClient.doResourceDelete(LDAPCONFIG_TYPE, &container.Resource)
}
