# Example
You can instantiate `*dynamodb.DynamoDB` and pass that as a parameter to all
methods connecting to DynamoDB, or as `unitTest` demonstrates, create your own
`type` and pass it along as a field.

## Test-compatible DynamoDB field
If you use `*dynamodb.DynamoDB` as a field, you will be unable to unit test it,
as documented in #88. Cast it instead as `dynamodbiface.DynamoDBAPI`:

```go
type ItemGetter struct {
		DynamoDB dynamodbiface.DynamoDBAPI
}
```

## Querying actual DynamoDB
You'll need an `*aws.Config` and `*session.Session` for these to work correctly:

```go
// Setup
var getter = new(ItemGetter)
var config *aws.Config = &aws.Config{Region: aws.String("us-west-2"),}
var sess *session.Session = session.NewSession(config)
var svc *dynamodb.DynamoDB = dynamodb.New()
getter.DynamoDB = dynamodbiface.DynamoDBAPI(svc)
// Finally
getter.DynamoDB.GetItem(/* ... */)
```

## Querying in tests
Construct a `fakeDynamoDB` and add the necessary methods for each of those
structs (custom ones for `ItemGetter` and [whatever methods you're using for
DynamoDB](https://github.com/aws/aws-sdk-go/blob/master/service/dynamodb/dynamodbiface/interface.go)),
and you're good to go!

```go
type fakeDynamoDB struct {
		dynamodbiface.DynamoDBAPI
}
var getter = new(ItemGetter)
getter.DynamoDB = &fakeDynamoDB{}
// And to run it (assuming you've mocked fakeDynamoDB.GetItem)
getter.DynamoDB.GetItem(/* ... */)
```

## Output
```
$ go test -tags example -cover
PASS
coverage: 100.0% of statements
ok		_/Users/shatil/workspace/aws-sdk-go/example/service/dynamodb/unitTest	0.008s
```
