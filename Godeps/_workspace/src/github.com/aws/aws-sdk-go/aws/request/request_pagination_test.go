package request_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/test/unit"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/s3"
)

var _ = unit.Imported

// Use DynamoDB methods for simplicity
func TestPagination(t *testing.T) {
	db := dynamodb.New(nil)
	tokens, pages, numPages, gotToEnd := []string{}, []string{}, 0, false

	reqNum := 0
	resps := []*dynamodb.ListTablesOutput{
		{TableNames: []*string{aws.String("Table1"), aws.String("Table2")}, LastEvaluatedTableName: aws.String("Table2")},
		{TableNames: []*string{aws.String("Table3"), aws.String("Table4")}, LastEvaluatedTableName: aws.String("Table4")},
		{TableNames: []*string{aws.String("Table5")}},
	}

	db.Handlers.Send.Clear() // mock sending
	db.Handlers.Unmarshal.Clear()
	db.Handlers.UnmarshalMeta.Clear()
	db.Handlers.ValidateResponse.Clear()
	db.Handlers.Build.PushBack(func(r *request.Request) {
		in := r.Params.(*dynamodb.ListTablesInput)
		if in == nil {
			tokens = append(tokens, "")
		} else if in.ExclusiveStartTableName != nil {
			tokens = append(tokens, *in.ExclusiveStartTableName)
		}
	})
	db.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = resps[reqNum]
		reqNum++
	})

	params := &dynamodb.ListTablesInput{Limit: aws.Int64(2)}
	err := db.ListTablesPages(params, func(p *dynamodb.ListTablesOutput, last bool) bool {
		numPages++
		for _, t := range p.TableNames {
			pages = append(pages, *t)
		}
		if last {
			if gotToEnd {
				assert.Fail(t, "last=true happened twice")
			}
			gotToEnd = true
		}
		return true
	})

	assert.Equal(t, []string{"Table2", "Table4"}, tokens)
	assert.Equal(t, []string{"Table1", "Table2", "Table3", "Table4", "Table5"}, pages)
	assert.Equal(t, 3, numPages)
	assert.True(t, gotToEnd)
	assert.Nil(t, err)
	assert.Nil(t, params.ExclusiveStartTableName)
}

// Use DynamoDB methods for simplicity
func TestPaginationEachPage(t *testing.T) {
	db := dynamodb.New(nil)
	tokens, pages, numPages, gotToEnd := []string{}, []string{}, 0, false

	reqNum := 0
	resps := []*dynamodb.ListTablesOutput{
		{TableNames: []*string{aws.String("Table1"), aws.String("Table2")}, LastEvaluatedTableName: aws.String("Table2")},
		{TableNames: []*string{aws.String("Table3"), aws.String("Table4")}, LastEvaluatedTableName: aws.String("Table4")},
		{TableNames: []*string{aws.String("Table5")}},
	}

	db.Handlers.Send.Clear() // mock sending
	db.Handlers.Unmarshal.Clear()
	db.Handlers.UnmarshalMeta.Clear()
	db.Handlers.ValidateResponse.Clear()
	db.Handlers.Build.PushBack(func(r *request.Request) {
		in := r.Params.(*dynamodb.ListTablesInput)
		if in == nil {
			tokens = append(tokens, "")
		} else if in.ExclusiveStartTableName != nil {
			tokens = append(tokens, *in.ExclusiveStartTableName)
		}
	})
	db.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = resps[reqNum]
		reqNum++
	})

	params := &dynamodb.ListTablesInput{Limit: aws.Int64(2)}
	req, _ := db.ListTablesRequest(params)
	err := req.EachPage(func(p interface{}, last bool) bool {
		numPages++
		for _, t := range p.(*dynamodb.ListTablesOutput).TableNames {
			pages = append(pages, *t)
		}
		if last {
			if gotToEnd {
				assert.Fail(t, "last=true happened twice")
			}
			gotToEnd = true
		}

		return true
	})

	assert.Equal(t, []string{"Table2", "Table4"}, tokens)
	assert.Equal(t, []string{"Table1", "Table2", "Table3", "Table4", "Table5"}, pages)
	assert.Equal(t, 3, numPages)
	assert.True(t, gotToEnd)
	assert.Nil(t, err)
}

// Use DynamoDB methods for simplicity
func TestPaginationEarlyExit(t *testing.T) {
	db := dynamodb.New(nil)
	numPages, gotToEnd := 0, false

	reqNum := 0
	resps := []*dynamodb.ListTablesOutput{
		{TableNames: []*string{aws.String("Table1"), aws.String("Table2")}, LastEvaluatedTableName: aws.String("Table2")},
		{TableNames: []*string{aws.String("Table3"), aws.String("Table4")}, LastEvaluatedTableName: aws.String("Table4")},
		{TableNames: []*string{aws.String("Table5")}},
	}

	db.Handlers.Send.Clear() // mock sending
	db.Handlers.Unmarshal.Clear()
	db.Handlers.UnmarshalMeta.Clear()
	db.Handlers.ValidateResponse.Clear()
	db.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = resps[reqNum]
		reqNum++
	})

	params := &dynamodb.ListTablesInput{Limit: aws.Int64(2)}
	err := db.ListTablesPages(params, func(p *dynamodb.ListTablesOutput, last bool) bool {
		numPages++
		if numPages == 2 {
			return false
		}
		if last {
			if gotToEnd {
				assert.Fail(t, "last=true happened twice")
			}
			gotToEnd = true
		}
		return true
	})

	assert.Equal(t, 2, numPages)
	assert.False(t, gotToEnd)
	assert.Nil(t, err)
}

func TestSkipPagination(t *testing.T) {
	client := s3.New(nil)
	client.Handlers.Send.Clear() // mock sending
	client.Handlers.Unmarshal.Clear()
	client.Handlers.UnmarshalMeta.Clear()
	client.Handlers.ValidateResponse.Clear()
	client.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = &s3.HeadBucketOutput{}
	})

	req, _ := client.HeadBucketRequest(&s3.HeadBucketInput{Bucket: aws.String("bucket")})

	numPages, gotToEnd := 0, false
	req.EachPage(func(p interface{}, last bool) bool {
		numPages++
		if last {
			gotToEnd = true
		}
		return true
	})
	assert.Equal(t, 1, numPages)
	assert.True(t, gotToEnd)
}

// Use S3 for simplicity
func TestPaginationTruncation(t *testing.T) {
	count := 0
	client := s3.New(nil)

	reqNum := &count
	resps := []*s3.ListObjectsOutput{
		{IsTruncated: aws.Bool(true), Contents: []*s3.Object{{Key: aws.String("Key1")}}},
		{IsTruncated: aws.Bool(true), Contents: []*s3.Object{{Key: aws.String("Key2")}}},
		{IsTruncated: aws.Bool(false), Contents: []*s3.Object{{Key: aws.String("Key3")}}},
		{IsTruncated: aws.Bool(true), Contents: []*s3.Object{{Key: aws.String("Key4")}}},
	}

	client.Handlers.Send.Clear() // mock sending
	client.Handlers.Unmarshal.Clear()
	client.Handlers.UnmarshalMeta.Clear()
	client.Handlers.ValidateResponse.Clear()
	client.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = resps[*reqNum]
		*reqNum++
	})

	params := &s3.ListObjectsInput{Bucket: aws.String("bucket")}

	results := []string{}
	err := client.ListObjectsPages(params, func(p *s3.ListObjectsOutput, last bool) bool {
		results = append(results, *p.Contents[0].Key)
		return true
	})

	assert.Equal(t, []string{"Key1", "Key2", "Key3"}, results)
	assert.Nil(t, err)

	// Try again without truncation token at all
	count = 0
	resps[1].IsTruncated = nil
	resps[2].IsTruncated = aws.Bool(true)
	results = []string{}
	err = client.ListObjectsPages(params, func(p *s3.ListObjectsOutput, last bool) bool {
		results = append(results, *p.Contents[0].Key)
		return true
	})

	assert.Equal(t, []string{"Key1", "Key2"}, results)
	assert.Nil(t, err)

}

// Benchmarks
var benchResps = []*dynamodb.ListTablesOutput{
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE"), aws.String("NXT")}, LastEvaluatedTableName: aws.String("NXT")},
	{TableNames: []*string{aws.String("TABLE")}},
}

var benchDb = func() *dynamodb.DynamoDB {
	db := dynamodb.New(nil)
	db.Handlers.Send.Clear() // mock sending
	db.Handlers.Unmarshal.Clear()
	db.Handlers.UnmarshalMeta.Clear()
	db.Handlers.ValidateResponse.Clear()
	return db
}

func BenchmarkCodegenIterator(b *testing.B) {
	reqNum := 0
	db := benchDb()
	db.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = benchResps[reqNum]
		reqNum++
	})

	input := &dynamodb.ListTablesInput{Limit: aws.Int64(2)}
	iter := func(fn func(*dynamodb.ListTablesOutput, bool) bool) error {
		page, _ := db.ListTablesRequest(input)
		for ; page != nil; page = page.NextPage() {
			page.Send()
			out := page.Data.(*dynamodb.ListTablesOutput)
			if result := fn(out, !page.HasNextPage()); page.Error != nil || !result {
				return page.Error
			}
		}
		return nil
	}

	for i := 0; i < b.N; i++ {
		reqNum = 0
		iter(func(p *dynamodb.ListTablesOutput, last bool) bool {
			return true
		})
	}
}

func BenchmarkEachPageIterator(b *testing.B) {
	reqNum := 0
	db := benchDb()
	db.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		r.Data = benchResps[reqNum]
		reqNum++
	})

	input := &dynamodb.ListTablesInput{Limit: aws.Int64(2)}
	for i := 0; i < b.N; i++ {
		reqNum = 0
		req, _ := db.ListTablesRequest(input)
		req.EachPage(func(p interface{}, last bool) bool {
			return true
		})
	}
}
