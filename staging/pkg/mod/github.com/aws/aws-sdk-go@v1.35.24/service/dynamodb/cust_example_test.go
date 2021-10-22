package dynamodb_test

import (
	"log"

	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

func ExampleDynamoDB_TransactWriteItems_transactionCanceledException() {
	client := dynamodb.New(unit.Session)

	_, err := client.TransactWriteItems(&dynamodb.TransactWriteItemsInput{})
	if err != nil {
		switch t := err.(type) {
		case *dynamodb.TransactionCanceledException:
			log.Fatalf("failed to write items: %s\n%v",
				t.Message(), t.CancellationReasons)
		default:
			log.Fatalf("failed to write items: %v", err)
		}
	}
}
