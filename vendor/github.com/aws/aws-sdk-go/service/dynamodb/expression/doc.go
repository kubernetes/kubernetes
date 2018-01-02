/*
Package expression provides types and functions to create Amazon DynamoDB
Expression strings, ExpressionAttributeNames maps, and ExpressionAttributeValues
maps.

Using the Package

The package represents the various DynamoDB Expressions as structs named
accordingly. For example, ConditionBuilder represents a DynamoDB Condition
Expression, an UpdateBuilder represents a DynamoDB Update Expression, and so on.
The following example shows a sample ConditionExpression and how to build an
equilvalent ConditionBuilder

  // Let :a be an ExpressionAttributeValue representing the string "No One You
  // Know"
  condExpr := "Artist = :a"
  condBuilder := expression.Name("Artist").Equal(expression.Value("No One You Know"))

In order to retrieve the formatted DynamoDB Expression strings, call the getter
methods on the Expression struct. To create the Expression struct, call the
Build() method on the Builder struct. Because some input structs, such as
QueryInput, can have multiple DynamoDB Expressions, multiple structs
representing various DynamoDB Expressions can be added to the Builder struct.
The following example shows a generic usage of the whole package.

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

The ExpressionAttributeNames and ExpressionAttributeValues member of the input
struct must always be assigned when using the Expression struct because all item
attribute names and values are aliased. That means that if the
ExpressionAttributeNames and ExpressionAttributeValues member is not assigned
with the corresponding Names() and Values() methods, the DynamoDB operation will
run into a logic error.
*/
package expression
