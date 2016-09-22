package client

const (
	KUBERNETES_STACK_UPGRADE_TYPE = "kubernetesStackUpgrade"
)

type KubernetesStackUpgrade struct {
	Resource

	Environment map[string]interface{} `json:"environment,omitempty" yaml:"environment,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Templates map[string]interface{} `json:"templates,omitempty" yaml:"templates,omitempty"`
}

type KubernetesStackUpgradeCollection struct {
	Collection
	Data []KubernetesStackUpgrade `json:"data,omitempty"`
}

type KubernetesStackUpgradeClient struct {
	rancherClient *RancherClient
}

type KubernetesStackUpgradeOperations interface {
	List(opts *ListOpts) (*KubernetesStackUpgradeCollection, error)
	Create(opts *KubernetesStackUpgrade) (*KubernetesStackUpgrade, error)
	Update(existing *KubernetesStackUpgrade, updates interface{}) (*KubernetesStackUpgrade, error)
	ById(id string) (*KubernetesStackUpgrade, error)
	Delete(container *KubernetesStackUpgrade) error
}

func newKubernetesStackUpgradeClient(rancherClient *RancherClient) *KubernetesStackUpgradeClient {
	return &KubernetesStackUpgradeClient{
		rancherClient: rancherClient,
	}
}

func (c *KubernetesStackUpgradeClient) Create(container *KubernetesStackUpgrade) (*KubernetesStackUpgrade, error) {
	resp := &KubernetesStackUpgrade{}
	err := c.rancherClient.doCreate(KUBERNETES_STACK_UPGRADE_TYPE, container, resp)
	return resp, err
}

func (c *KubernetesStackUpgradeClient) Update(existing *KubernetesStackUpgrade, updates interface{}) (*KubernetesStackUpgrade, error) {
	resp := &KubernetesStackUpgrade{}
	err := c.rancherClient.doUpdate(KUBERNETES_STACK_UPGRADE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *KubernetesStackUpgradeClient) List(opts *ListOpts) (*KubernetesStackUpgradeCollection, error) {
	resp := &KubernetesStackUpgradeCollection{}
	err := c.rancherClient.doList(KUBERNETES_STACK_UPGRADE_TYPE, opts, resp)
	return resp, err
}

func (c *KubernetesStackUpgradeClient) ById(id string) (*KubernetesStackUpgrade, error) {
	resp := &KubernetesStackUpgrade{}
	err := c.rancherClient.doById(KUBERNETES_STACK_UPGRADE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *KubernetesStackUpgradeClient) Delete(container *KubernetesStackUpgrade) error {
	return c.rancherClient.doResourceDelete(KUBERNETES_STACK_UPGRADE_TYPE, &container.Resource)
}
