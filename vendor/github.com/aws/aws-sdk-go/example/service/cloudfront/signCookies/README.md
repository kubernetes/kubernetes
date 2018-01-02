# Example

This example shows how the CloudFront CookieSigner can be used to generate signed cookies to provided short term access to restricted resourced fronted by CloudFront.

# Usage
Makes a request for object using CloudFront cookie signing, and outputs the contents of the object to stdout.

```sh
go run -tags example signCookies.go -file <privkey file>  -id <keyId> -r <resource pattern> -g <object to get>
```


