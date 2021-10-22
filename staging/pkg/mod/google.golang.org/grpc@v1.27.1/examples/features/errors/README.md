# Description

This example demonstrates the use of status details in grpc errors.

# Run the sample code

Run the server:

```sh
$ go run ./server/main.go
```
Then run the client in another terminal:

```sh
$ go run ./client/main.go
```

It should succeed and print the greeting it received from the server.
Then run the client again:

```sh
$ go run ./client/main.go
```

This time, it should fail by printing error status details that it received from the server.
