// +build integration

// Package performance contains shared step definitions that are used for performance testing
package performance

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
)

// benchmarkLogger handles all benchmark logging
type benchmarkLogger struct {
	outputer
}

// logger interface that handles any logging to an output
type logger interface {
	log(key string, data map[string]interface{}) error
}

// init initializes the logger and uses dependency injection for the
// outputer
func newBenchmarkLogger(output string) (*benchmarkLogger, error) {
	b := &benchmarkLogger{}
	switch output {
	case "dynamodb":
		region := os.Getenv("AWS_TESTING_REGION")
		if region == "" {
			return b, errors.New("No region specified. Please export AWS_TESTING_REGION")
		}

		table := os.Getenv("AWS_TESTING_DB_TABLE")
		if table == "" {
			return b, errors.New("No table specified. Please export AWS_TESTING_DB_TABLE")
		}
		b.outputer = newDynamodbOut(table, region)
	case "stdout":
		b.outputer = stdout{}
	default:
		return b, errors.New("Unsupported outputer")
	}
	return b, nil
}

type record struct {
	Key  string
	Data interface{}
}

// log calls the output command and building a data structure
// to pass into its output formatter
func (b benchmarkLogger) log(key, data interface{}) error {
	formatData := record{
		Key:  fmt.Sprintf("%d-%v", time.Now().Unix(), key.(string)),
		Data: data,
	}

	return b.output(formatData)
}

// outputer is a simple interface that'll handle output
// to whatever system like dynamodb or stdout
type outputer interface {
	output(record) error
}

// dyanmodbOut handles simple writes to dynamodb
type dynamodbOut struct {
	table  string // table to write to in dynamodb
	region string
	db     *dynamodb.DynamoDB // the dynamodb
}

// init initializes dynamodbOut
func newDynamodbOut(table, region string) *dynamodbOut {
	out := dynamodbOut{
		table:  table,
		region: region,
	}

	out.db = dynamodb.New(
		unit.Session,
		&aws.Config{Region: &out.region},
	)
	return &out
}

// output just writes to dynamodb
func (out dynamodbOut) output(data record) error {
	input := &dynamodb.PutItemInput{
		TableName: aws.String(out.table),
	}

	item, err := dynamodbattribute.ConvertToMap(data)
	if err != nil {
		return err
	}

	input.Item = item
	_, err = out.db.PutItem(input)
	return err
}

// stdout handles writes to stdout
type stdout struct{}

// output expects key value data to print to stdout
func (out stdout) output(data record) error {
	item, err := dynamodbattribute.ConvertToMap(data.Data)
	if err != nil {
		return err
	}
	fmt.Println(item)
	return nil
}
