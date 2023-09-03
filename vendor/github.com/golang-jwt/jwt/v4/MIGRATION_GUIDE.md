## Migration Guide (v4.0.0)

Starting from [v4.0.0](https://github.com/golang-jwt/jwt/releases/tag/v4.0.0), the import path will be:

    "github.com/golang-jwt/jwt/v4"

The `/v4` version will be backwards compatible with existing `v3.x.y` tags in this repo, as well as 
`github.com/dgrijalva/jwt-go`. For most users this should be a drop-in replacement, if you're having 
troubles migrating, please open an issue.

You can replace all occurrences of `github.com/dgrijalva/jwt-go` or `github.com/golang-jwt/jwt` with `github.com/golang-jwt/jwt/v4`, either manually or by using tools such as `sed` or `gofmt`.

And then you'd typically run:

```
go get github.com/golang-jwt/jwt/v4
go mod tidy
```

## Older releases (before v3.2.0)

The original migration guide for older releases can be found at https://github.com/dgrijalva/jwt-go/blob/master/MIGRATION_GUIDE.md.
