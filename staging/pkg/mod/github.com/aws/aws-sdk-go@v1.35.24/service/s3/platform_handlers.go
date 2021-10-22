// +build !go1.6

package s3

import "github.com/aws/aws-sdk-go/aws/request"

func platformRequestHandlers(r *request.Request) {
}
