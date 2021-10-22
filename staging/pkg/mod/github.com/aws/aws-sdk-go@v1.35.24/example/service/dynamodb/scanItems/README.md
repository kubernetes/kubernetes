# Example

`scanItems` is an example how to use Amazon DynamoDB's Scan API operation with the SDK's `dynamodbattributes.UnmarshalListOfMaps` to unmarshal the Scan response's `Items` `[]map[string]*dynamodb.AttributeValue` field. This unmarshaler can be used with all `[]map[string]*dynamodb.AttributeValue` type fields.

## Go Type

The `Item` time will be used by the example to unmarshal the DynamoDB table's items to.

```go
type Item struct {
	Key  int
	Desc string
	Data map[string]interface{}
}
```
Use Go tags to define what the name is of the attribute in your DynamoDB table. See [AWS SDK for Go API Reference: Marshal](https://docs.aws.amazon.com/sdk-for-go/api/service/dynamodb/dynamodbattribute/#Marshal) for more information.

In DynamoDB the structure of the item to be returned will be:
```json
{
  "Data": {
    "Value 1": "abc",
    "Value 2": 1234567890
  },
  "Desc": "First ddb item",
  "Key": 1
}
```

## Usage

`go run -tags example scanItems.go -table "<table_name>" -region "<optional_region>"`

## Output

```
0: Key: 123, Desc: An item in the DynamoDB table
	Num Data Values: 0
1: Key: 2, Desc: Second ddb item
	Num Data Values: 2
	- "A Field": 123
	- "Another Field": abc
2: Key: 1, Desc: First ddb item
	Num Data Values: 2
	- "Value 1": abc
	- "Value 2": 1234567890
```
