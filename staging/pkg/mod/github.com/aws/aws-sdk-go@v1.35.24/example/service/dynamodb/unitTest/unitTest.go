// +build example

// Package unitTest demonstrates how to unit test, without needing to pass a
// connector to every function, code that uses DynamoDB.
package unitTest

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbiface"
)

// ItemGetter can be assigned a DynamoDB connector like:
//	svc := dynamodb.DynamoDB(sess)
//	getter.DynamoDB = dynamodbiface.DynamoDBAPI(svc)
type ItemGetter struct {
	DynamoDB dynamodbiface.DynamoDBAPI
}

// Get a value from a DynamoDB table containing entries like:
// {"id": "my primary key", "value": "valuable value"}
func (ig *ItemGetter) Get(id string) (value string) {
	var input = &dynamodb.GetItemInput{
		Key: map[string]*dynamodb.AttributeValue{
			"id": {
				S: aws.String(id),
			},
		},
		TableName: aws.String("my_table"),
		AttributesToGet: []*string{
			aws.String("value"),
		},
	}
	if output, err := ig.DynamoDB.GetItem(input); err == nil {
		if _, ok := output.Item["value"]; ok {
			dynamodbattribute.Unmarshal(output.Item["value"], &value)
		}
	}
	return
}
