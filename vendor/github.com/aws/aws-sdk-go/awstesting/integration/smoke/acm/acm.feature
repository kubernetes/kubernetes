#language en
@acm @client
Feature: AWS Certificate Manager

	Scenario: Making a request
		When I call the "ListCertificates" API
		Then the request should be successful

	Scenario: Handling errors
		When I attempt to call the "GetCertificate" API with:
			| CertificateArn |  arn:aws:acm:region:123456789012:certificate/12345678-1234-1234-1234-123456789012 |
		Then I expect the response error code to be "ResourceNotFoundException"
		And I expect the response error message not be empty

