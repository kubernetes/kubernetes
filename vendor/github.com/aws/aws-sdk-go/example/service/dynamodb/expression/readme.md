# Example

`scan` is an example how to use Amazon DynamoDB's `expression` package to fill
the member fields of Amazon DynamoDB's Operation input types.

## Representing DynamoDB Expressions

In the example, the variable `filt` represents a `FilterExpression`. Note that
DynamoDB item attributes are represented using the function `Name()` and
DynamoDB item values are similarly represented using the function `Value()`. In
this context, the string `"Artist"` represents the name of the item attribute
that we want to evaluate and the string `"No One You Know"` represents the value
we want to evaluate the item attribute against. The relationship between the two
[operands](http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html#Expressions.OperatorsAndFunctions.Syntax)
are specified using the method `Equal()`.

Similarly, the variable `proj` represents a `ProjectionExpression`. The list of
item attribute names comprising the `ProjectionExpression` are specified as
arguments to the function `NamesList()`. The `expression` package utilizes the
type safety of Go and if an item value were to be used as an argument to the
function `NamesList()`, a compile time error is returned. The pattern of
representing DynamoDB Expressions by indicating relationships between `operands`
with functions is consistent throughout the whole `expression` package.

```go
filt := expression.Name("Artist").Equal(expression.Value("No One You Know"))
// let :a be an ExpressionAttributeValue representing the string "No One You Know"
// equivalent FilterExpression: "Artist = :a"

proj := expression.NamesList(expression.Name("SongTitle"), expression.Name("AlbumTitle"))
// equivalent ProjectionExpression: "SongTitle, AlbumTitle"
```

## Creating an `Expression`

In the example, the variable `expr` is an instance of an `Expression` type. An
`Expression` is built using a builder pattern. First, a new `Builder` is
initialized by the `NewBuilder()` function. Then, types representing DynamoDB
Expressions are added to the `Builder` by methods `WithFilter()` and
`WithProjection()`. The `Build()` method returns an instance of an `Expression`
and an error. The error will be either an `InvalidParameterError` or an
`UnsetParameterError`.

```go
filt := expression.Name("Artist").Equal(expression.Value("No One You Know"))
proj := expression.NamesList(expression.Name("SongTitle"), expression.Name("AlbumTitle"))

expr, err := expression.NewBuilder().WithFilter(filt).WithProjection(proj).Build()
if err != nil {
  fmt.Println(err)
}
```

## Filling in the fields of a DynamoDB `Scan` API

In the example, the getter methods of the `Expression` type is used to get the
formatted DynamoDB Expression strings. The `ExpressionAttributeNames` and
`ExpressionAttributeValues` member field of the DynamoDB API must always be
assigned when using an `Expression` since all item attribute names and values
are aliased. That means that if the `ExpressionAttributeNames` and
`ExpressionAttributeValues` member is not assigned with the corresponding
`Names()` and `Values()` methods, the DynamoDB operation will run into a logic
error.

```go
filt := expression.Name("Artist").Equal(expression.Value("No One You Know"))
proj := expression.NamesList(expression.Name("SongTitle"), expression.Name("AlbumTitle"))
expr, err := expression.NewBuilder().WithFilter(filt).WithProjection(proj).Build()
if err != nil {
  fmt.Println(err)
}

input := &dynamodb.ScanInput{
  ExpressionAttributeNames:  expr.Names(),
  ExpressionAttributeValues: expr.Values(),
  FilterExpression:          expr.Filter(),
  ProjectionExpression:      expr.Projection(),
  TableName:                 aws.String("Music"),
}
```

## Usage

`go run -tags example scan.go -table "<table_name>" -region "<optional_region>"`

## Output

```
{
	Count: #SomeNumber,
	Items: [{
		AlbumTitle: {
			#SomeAlbumTitle
		},
		SongTitle: {
			#SomeSongTitle
		}
	}],
	...
	ScannedCount: #SomeNumber,
}
```
