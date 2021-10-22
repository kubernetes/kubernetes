package simpledb

import "github.com/aws/aws-sdk-go/aws/client"

func init() {
	initClient = func(c *client.Client) {
		// SimpleDB uses custom error unmarshaling logic
		c.Handlers.UnmarshalError.Clear()
		c.Handlers.UnmarshalError.PushBack(unmarshalError)
	}
}
