package client

import "fmt"

type Client struct{}

func (c *Client) Greet(in GreetInput) string {
	return fmt.Sprintf("Hello, %s!", in.Name)
}

type GreetInput struct {
	Name string
}
