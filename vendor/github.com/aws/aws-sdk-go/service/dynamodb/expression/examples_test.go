package expression_test

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/expression"
)

// Using Projection Expression
//
// This example queries items in the Music table. The table has a partition key and
// sort key (Artist and SongTitle), but this query only specifies the partition key
// value. It returns song titles by the artist named "No One You Know".
func ExampleBuilder_WithProjection() {
	svc := dynamodb.New(session.New())

	// Construct the Key condition builder
	keyCond := expression.Key("Artist").Equal(expression.Value("No One You Know"))

	// Create the project expression builder with a names list.
	proj := expression.NamesList(expression.Name("SongTitle"))

	// Combine the key condition, and projection together as a DynamoDB expression
	// builder.
	expr, err := expression.NewBuilder().
		WithKeyCondition(keyCond).
		WithProjection(proj).
		Build()
	if err != nil {
		fmt.Println(err)
	}

	// Use the built expression to populate the DynamoDB Query's API input
	// parameters.
	input := &dynamodb.QueryInput{
		ExpressionAttributeValues: expr.Values(),
		KeyConditionExpression:    expr.KeyCondition(),
		ProjectionExpression:      expr.Projection(),
		TableName:                 aws.String("Music"),
	}

	result, err := svc.Query(input)
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case dynamodb.ErrCodeProvisionedThroughputExceededException:
				fmt.Println(dynamodb.ErrCodeProvisionedThroughputExceededException, aerr.Error())
			case dynamodb.ErrCodeResourceNotFoundException:
				fmt.Println(dynamodb.ErrCodeResourceNotFoundException, aerr.Error())
			case dynamodb.ErrCodeInternalServerError:
				fmt.Println(dynamodb.ErrCodeInternalServerError, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}
		return
	}

	fmt.Println(result)
}

// Using Key Condition Expression
//
// This example queries items in the Music table. The table has a partition key and
// sort key (Artist and SongTitle), but this query only specifies the partition key
// value. It returns song titles by the artist named "No One You Know".
func ExampleBuilder_WithKeyCondition() {
	svc := dynamodb.New(session.New())

	// Construct the Key condition builder
	keyCond := expression.Key("Artist").Equal(expression.Value("No One You Know"))

	// Create the project expression builder with a names list.
	proj := expression.NamesList(expression.Name("SongTitle"))

	// Combine the key condition, and projection together as a DynamoDB expression
	// builder.
	expr, err := expression.NewBuilder().
		WithKeyCondition(keyCond).
		WithProjection(proj).
		Build()
	if err != nil {
		fmt.Println(err)
	}

	// Use the built expression to populate the DynamoDB Query's API input
	// parameters.
	input := &dynamodb.QueryInput{
		ExpressionAttributeValues: expr.Values(),
		KeyConditionExpression:    expr.KeyCondition(),
		ProjectionExpression:      expr.Projection(),
		TableName:                 aws.String("Music"),
	}

	result, err := svc.Query(input)
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case dynamodb.ErrCodeProvisionedThroughputExceededException:
				fmt.Println(dynamodb.ErrCodeProvisionedThroughputExceededException, aerr.Error())
			case dynamodb.ErrCodeResourceNotFoundException:
				fmt.Println(dynamodb.ErrCodeResourceNotFoundException, aerr.Error())
			case dynamodb.ErrCodeInternalServerError:
				fmt.Println(dynamodb.ErrCodeInternalServerError, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}
		return
	}

	fmt.Println(result)
}

// Using Filter Expression
//
// This example scans the entire Music table, and then narrows the results to songs
// by the artist "No One You Know". For each item, only the album title and song title
// are returned.
func ExampleBuilder_WithFilter() {
	svc := dynamodb.New(session.New())

	// Construct the filter builder with a name and value.
	filt := expression.Name("Artist").Equal(expression.Value("No One You Know"))

	// Create the names list projection of names to project.
	proj := expression.NamesList(
		expression.Name("AlbumTitle"),
		expression.Name("SongTitle"),
	)

	// Using the filter and projections create a DynamoDB expression from the two.
	expr, err := expression.NewBuilder().
		WithFilter(filt).
		WithProjection(proj).
		Build()
	if err != nil {
		fmt.Println(err)
	}

	// Use the built expression to populate the DynamoDB Scan API input parameters.
	input := &dynamodb.ScanInput{
		ExpressionAttributeNames:  expr.Names(),
		ExpressionAttributeValues: expr.Values(),
		FilterExpression:          expr.Filter(),
		ProjectionExpression:      expr.Projection(),
		TableName:                 aws.String("Music"),
	}

	result, err := svc.Scan(input)
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case dynamodb.ErrCodeProvisionedThroughputExceededException:
				fmt.Println(dynamodb.ErrCodeProvisionedThroughputExceededException, aerr.Error())
			case dynamodb.ErrCodeResourceNotFoundException:
				fmt.Println(dynamodb.ErrCodeResourceNotFoundException, aerr.Error())
			case dynamodb.ErrCodeInternalServerError:
				fmt.Println(dynamodb.ErrCodeInternalServerError, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}
		return
	}

	fmt.Println(result)
}

// Using Update Expression
//
// This example updates an item in the Music table. It adds a new attribute (Year) and
// modifies the AlbumTitle attribute.  All of the attributes in the item, as they appear
// after the update, are returned in the response.
func ExampleBuilder_WithUpdate() {
	svc := dynamodb.New(session.New())

	// Create an update to set two fields in the table.
	update := expression.Set(
		expression.Name("Year"),
		expression.Value(2015),
	).Set(
		expression.Name("AlbumTitle"),
		expression.Value("Louder Than Ever"),
	)

	// Create the DynamoDB expression from the Update.
	expr, err := expression.NewBuilder().
		WithUpdate(update).
		Build()

	// Use the built expression to populate the DynamoDB UpdateItem API
	// input parameters.
	input := &dynamodb.UpdateItemInput{
		ExpressionAttributeNames:  expr.Names(),
		ExpressionAttributeValues: expr.Values(),
		Key: map[string]*dynamodb.AttributeValue{
			"Artist": {
				S: aws.String("Acme Band"),
			},
			"SongTitle": {
				S: aws.String("Happy Day"),
			},
		},
		ReturnValues:     aws.String("ALL_NEW"),
		TableName:        aws.String("Music"),
		UpdateExpression: expr.Update(),
	}

	result, err := svc.UpdateItem(input)
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case dynamodb.ErrCodeConditionalCheckFailedException:
				fmt.Println(dynamodb.ErrCodeConditionalCheckFailedException, aerr.Error())
			case dynamodb.ErrCodeProvisionedThroughputExceededException:
				fmt.Println(dynamodb.ErrCodeProvisionedThroughputExceededException, aerr.Error())
			case dynamodb.ErrCodeResourceNotFoundException:
				fmt.Println(dynamodb.ErrCodeResourceNotFoundException, aerr.Error())
			case dynamodb.ErrCodeItemCollectionSizeLimitExceededException:
				fmt.Println(dynamodb.ErrCodeItemCollectionSizeLimitExceededException, aerr.Error())
			case dynamodb.ErrCodeInternalServerError:
				fmt.Println(dynamodb.ErrCodeInternalServerError, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}
		return
	}

	fmt.Println(result)
}

// Using Condition Expression
//
// This example deletes an item from the Music table if the rating is lower than
// 7.
func ExampleBuilder_WithCondition() {
	svc := dynamodb.New(session.New())

	// Create a condition where the Rating field must be less than 7.
	cond := expression.Name("Rating").LessThan(expression.Value(7))

	// Create a DynamoDB expression from the condition.
	expr, err := expression.NewBuilder().
		WithCondition(cond).
		Build()
	if err != nil {
		fmt.Println(err)
	}

	// Use the built expression to populate the DeleteItem API operation with the
	// condition expression.
	input := &dynamodb.DeleteItemInput{
		Key: map[string]*dynamodb.AttributeValue{
			"Artist": {
				S: aws.String("No One You Know"),
			},
			"SongTitle": {
				S: aws.String("Scared of My Shadow"),
			},
		},
		ExpressionAttributeNames:  expr.Names(),
		ExpressionAttributeValues: expr.Values(),
		ConditionExpression:       expr.Condition(),
		TableName:                 aws.String("Music"),
	}

	result, err := svc.DeleteItem(input)
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case dynamodb.ErrCodeConditionalCheckFailedException:
				fmt.Println(dynamodb.ErrCodeConditionalCheckFailedException, aerr.Error())
			case dynamodb.ErrCodeProvisionedThroughputExceededException:
				fmt.Println(dynamodb.ErrCodeProvisionedThroughputExceededException, aerr.Error())
			case dynamodb.ErrCodeResourceNotFoundException:
				fmt.Println(dynamodb.ErrCodeResourceNotFoundException, aerr.Error())
			case dynamodb.ErrCodeItemCollectionSizeLimitExceededException:
				fmt.Println(dynamodb.ErrCodeItemCollectionSizeLimitExceededException, aerr.Error())
			case dynamodb.ErrCodeInternalServerError:
				fmt.Println(dynamodb.ErrCodeInternalServerError, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}
		return
	}

	fmt.Println(result)
}
