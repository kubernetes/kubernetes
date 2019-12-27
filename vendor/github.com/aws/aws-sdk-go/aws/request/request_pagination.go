package request

import (
	"reflect"
	"sync/atomic"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
)

// A Pagination provides paginating of SDK API operations which are paginatable.
// Generally you should not use this type directly, but use the "Pages" API
// operations method to automatically perform pagination for you. Such as,
// "S3.ListObjectsPages", and "S3.ListObjectsPagesWithContext" methods.
//
// Pagination differs from a Paginator type in that pagination is the type that
// does the pagination between API operations, and Paginator defines the
// configuration that will be used per page request.
//
//     cont := true
//     for p.Next() && cont {
//         data := p.Page().(*s3.ListObjectsOutput)
//         // process the page's data
//     }
//     return p.Err()
//
// See service client API operation Pages methods for examples how the SDK will
// use the Pagination type.
type Pagination struct {
	// Function to return a Request value for each pagination request.
	// Any configuration or handlers that need to be applied to the request
	// prior to getting the next page should be done here before the request
	// returned.
	//
	// NewRequest should always be built from the same API operations. It is
	// undefined if different API operations are returned on subsequent calls.
	NewRequest func() (*Request, error)
	// EndPageOnSameToken, when enabled, will allow the paginator to stop on
	// token that are the same as its previous tokens.
	EndPageOnSameToken bool

	started    bool
	prevTokens []interface{}
	nextTokens []interface{}

	err     error
	curPage interface{}
}

// HasNextPage will return true if Pagination is able to determine that the API
// operation has additional pages. False will be returned if there are no more
// pages remaining.
//
// Will always return true if Next has not been called yet.
func (p *Pagination) HasNextPage() bool {
	if !p.started {
		return true
	}

	hasNextPage := len(p.nextTokens) != 0
	if p.EndPageOnSameToken {
		return hasNextPage && !awsutil.DeepEqual(p.nextTokens, p.prevTokens)
	}
	return hasNextPage
}

// Err returns the error Pagination encountered when retrieving the next page.
func (p *Pagination) Err() error {
	return p.err
}

// Page returns the current page. Page should only be called after a successful
// call to Next. It is undefined what Page will return if Page is called after
// Next returns false.
func (p *Pagination) Page() interface{} {
	return p.curPage
}

// Next will attempt to retrieve the next page for the API operation. When a page
// is retrieved true will be returned. If the page cannot be retrieved, or there
// are no more pages false will be returned.
//
// Use the Page method to retrieve the current page data. The data will need
// to be cast to the API operation's output type.
//
// Use the Err method to determine if an error occurred if Page returns false.
func (p *Pagination) Next() bool {
	if !p.HasNextPage() {
		return false
	}

	req, err := p.NewRequest()
	if err != nil {
		p.err = err
		return false
	}

	if p.started {
		for i, intok := range req.Operation.InputTokens {
			awsutil.SetValueAtPath(req.Params, intok, p.nextTokens[i])
		}
	}
	p.started = true

	err = req.Send()
	if err != nil {
		p.err = err
		return false
	}

	p.prevTokens = p.nextTokens
	p.nextTokens = req.nextPageTokens()
	p.curPage = req.Data

	return true
}

// A Paginator is the configuration data that defines how an API operation
// should be paginated. This type is used by the API service models to define
// the generated pagination config for service APIs.
//
// The Pagination type is what provides iterating between pages of an API. It
// is only used to store the token metadata the SDK should use for performing
// pagination.
type Paginator struct {
	InputTokens     []string
	OutputTokens    []string
	LimitToken      string
	TruncationToken string
}

// nextPageTokens returns the tokens to use when asking for the next page of data.
func (r *Request) nextPageTokens() []interface{} {
	if r.Operation.Paginator == nil {
		return nil
	}
	if r.Operation.TruncationToken != "" {
		tr, _ := awsutil.ValuesAtPath(r.Data, r.Operation.TruncationToken)
		if len(tr) == 0 {
			return nil
		}

		switch v := tr[0].(type) {
		case *bool:
			if !aws.BoolValue(v) {
				return nil
			}
		case bool:
			if !v {
				return nil
			}
		}
	}

	tokens := []interface{}{}
	tokenAdded := false
	for _, outToken := range r.Operation.OutputTokens {
		vs, _ := awsutil.ValuesAtPath(r.Data, outToken)
		if len(vs) == 0 {
			tokens = append(tokens, nil)
			continue
		}
		v := vs[0]

		switch tv := v.(type) {
		case *string:
			if len(aws.StringValue(tv)) == 0 {
				tokens = append(tokens, nil)
				continue
			}
		case string:
			if len(tv) == 0 {
				tokens = append(tokens, nil)
				continue
			}
		}

		tokenAdded = true
		tokens = append(tokens, v)
	}
	if !tokenAdded {
		return nil
	}

	return tokens
}

// Ensure a deprecated item is only logged once instead of each time its used.
func logDeprecatedf(logger aws.Logger, flag *int32, msg string) {
	if logger == nil {
		return
	}
	if atomic.CompareAndSwapInt32(flag, 0, 1) {
		logger.Log(msg)
	}
}

var (
	logDeprecatedHasNextPage int32
	logDeprecatedNextPage    int32
	logDeprecatedEachPage    int32
)

// HasNextPage returns true if this request has more pages of data available.
//
// Deprecated Use Pagination type for configurable pagination of API operations
func (r *Request) HasNextPage() bool {
	logDeprecatedf(r.Config.Logger, &logDeprecatedHasNextPage,
		"Request.HasNextPage deprecated. Use Pagination type for configurable pagination of API operations")

	return len(r.nextPageTokens()) > 0
}

// NextPage returns a new Request that can be executed to return the next
// page of result data. Call .Send() on this request to execute it.
//
// Deprecated Use Pagination type for configurable pagination of API operations
func (r *Request) NextPage() *Request {
	logDeprecatedf(r.Config.Logger, &logDeprecatedNextPage,
		"Request.NextPage deprecated. Use Pagination type for configurable pagination of API operations")

	tokens := r.nextPageTokens()
	if len(tokens) == 0 {
		return nil
	}

	data := reflect.New(reflect.TypeOf(r.Data).Elem()).Interface()
	nr := New(r.Config, r.ClientInfo, r.Handlers, r.Retryer, r.Operation, awsutil.CopyOf(r.Params), data)
	for i, intok := range nr.Operation.InputTokens {
		awsutil.SetValueAtPath(nr.Params, intok, tokens[i])
	}
	return nr
}

// EachPage iterates over each page of a paginated request object. The fn
// parameter should be a function with the following sample signature:
//
//   func(page *T, lastPage bool) bool {
//       return true // return false to stop iterating
//   }
//
// Where "T" is the structure type matching the output structure of the given
// operation. For example, a request object generated by
// DynamoDB.ListTablesRequest() would expect to see dynamodb.ListTablesOutput
// as the structure "T". The lastPage value represents whether the page is
// the last page of data or not. The return value of this function should
// return true to keep iterating or false to stop.
//
// Deprecated Use Pagination type for configurable pagination of API operations
func (r *Request) EachPage(fn func(data interface{}, isLastPage bool) (shouldContinue bool)) error {
	logDeprecatedf(r.Config.Logger, &logDeprecatedEachPage,
		"Request.EachPage deprecated. Use Pagination type for configurable pagination of API operations")

	for page := r; page != nil; page = page.NextPage() {
		if err := page.Send(); err != nil {
			return err
		}
		if getNextPage := fn(page.Data, !page.HasNextPage()); !getNextPage {
			return page.Error
		}
	}

	return nil
}
