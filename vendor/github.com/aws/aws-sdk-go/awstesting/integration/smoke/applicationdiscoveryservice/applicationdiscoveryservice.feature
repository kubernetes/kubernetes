#language en
@applicationdiscoveryservice @client
Feature: AWS Application Discovery Service

	Scenario: Making a request
		When I call the "DescribeAgents" API
		Then the request should be successful

