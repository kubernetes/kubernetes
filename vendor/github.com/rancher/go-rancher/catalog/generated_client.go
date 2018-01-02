package catalog

type RancherClient struct {
	RancherBaseClient

	ApiVersion      ApiVersionOperations
	Question        QuestionOperations
	Template        TemplateOperations
	TemplateVersion TemplateVersionOperations
	Catalog         CatalogOperations
	Error           ErrorOperations
}

func constructClient(rancherBaseClient *RancherBaseClientImpl) *RancherClient {
	client := &RancherClient{
		RancherBaseClient: rancherBaseClient,
	}

	client.ApiVersion = newApiVersionClient(client)
	client.Question = newQuestionClient(client)
	client.Template = newTemplateClient(client)
	client.TemplateVersion = newTemplateVersionClient(client)
	client.Catalog = newCatalogClient(client)
	client.Error = newErrorClient(client)

	return client
}

func NewRancherClient(opts *ClientOpts) (*RancherClient, error) {
	rancherBaseClient := &RancherBaseClientImpl{
		Types: map[string]Schema{},
	}
	client := constructClient(rancherBaseClient)

	err := setupRancherBaseClient(rancherBaseClient, opts)
	if err != nil {
		return nil, err
	}

	return client, nil
}
