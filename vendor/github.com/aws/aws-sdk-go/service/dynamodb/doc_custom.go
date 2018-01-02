/*
AttributeValue Marshaling and Unmarshaling Helpers

Utility helpers to marshal and unmarshal AttributeValue to and
from Go types can be found in the dynamodbattribute sub package. This package
provides has specialized functions for the common ways of working with
AttributeValues. Such as map[string]*AttributeValue, []*AttributeValue, and
directly with *AttributeValue. This is helpful for marshaling Go types for API
operations such as PutItem, and unmarshaling Query and Scan APIs' responses.

See the dynamodbattribute package documentation for more information.
https://docs.aws.amazon.com/sdk-for-go/api/service/dynamodb/dynamodbattribute/

Expression Builders

The expression package provides utility types and functions to build DynamoDB
expression for type safe construction of API ExpressionAttributeNames, and
ExpressionAttribute Values.

The package represents the various DynamoDB Expressions as structs named
accordingly. For example, ConditionBuilder represents a DynamoDB Condition
Expression, an UpdateBuilder represents a DynamoDB Update Expression, and so on.

See the expression package documentation for more information.
https://docs.aws.amazon.com/sdk-for-go/api/service/dynamodb/expression/
*/
package dynamodb
