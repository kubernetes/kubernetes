package jwt_test

// Example HTTP auth using asymmetric crypto/RSA keys
// This is based on a (now outdated) example at https://gist.github.com/cryptix/45c33ecf0ae54828e63b

import (
	"bytes"
	"crypto/rsa"
	"fmt"
	"github.com/dgrijalva/jwt-go"
	"github.com/dgrijalva/jwt-go/request"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// location of the files used for signing and verification
const (
	privKeyPath = "test/sample_key"     // openssl genrsa -out app.rsa keysize
	pubKeyPath  = "test/sample_key.pub" // openssl rsa -in app.rsa -pubout > app.rsa.pub
)

var (
	verifyKey  *rsa.PublicKey
	signKey    *rsa.PrivateKey
	serverPort int
	// storing sample username/password pairs
	// don't do this on a real server
	users = map[string]string{
		"test": "known",
	}
)

// read the key files before starting http handlers
func init() {
	signBytes, err := ioutil.ReadFile(privKeyPath)
	fatal(err)

	signKey, err = jwt.ParseRSAPrivateKeyFromPEM(signBytes)
	fatal(err)

	verifyBytes, err := ioutil.ReadFile(pubKeyPath)
	fatal(err)

	verifyKey, err = jwt.ParseRSAPublicKeyFromPEM(verifyBytes)
	fatal(err)

	http.HandleFunc("/authenticate", authHandler)
	http.HandleFunc("/restricted", restrictedHandler)

	// Setup listener
	listener, err := net.ListenTCP("tcp", &net.TCPAddr{})
	serverPort = listener.Addr().(*net.TCPAddr).Port

	log.Println("Listening...")
	go func() {
		fatal(http.Serve(listener, nil))
	}()
}

var start func()

func fatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// Define some custom types were going to use within our tokens
type CustomerInfo struct {
	Name string
	Kind string
}

type CustomClaimsExample struct {
	*jwt.StandardClaims
	TokenType string
	CustomerInfo
}

func Example_getTokenViaHTTP() {
	// See func authHandler for an example auth handler that produces a token
	res, err := http.PostForm(fmt.Sprintf("http://localhost:%v/authenticate", serverPort), url.Values{
		"user": {"test"},
		"pass": {"known"},
	})
	if err != nil {
		fatal(err)
	}

	if res.StatusCode != 200 {
		fmt.Println("Unexpected status code", res.StatusCode)
	}

	// Read the token out of the response body
	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()
	tokenString := strings.TrimSpace(buf.String())

	// Parse the token
	token, err := jwt.ParseWithClaims(tokenString, &CustomClaimsExample{}, func(token *jwt.Token) (interface{}, error) {
		// since we only use the one private key to sign the tokens,
		// we also only use its public counter part to verify
		return verifyKey, nil
	})
	fatal(err)

	claims := token.Claims.(*CustomClaimsExample)
	fmt.Println(claims.CustomerInfo.Name)

	//Output: test
}

func Example_useTokenViaHTTP() {

	// Make a sample token
	// In a real world situation, this token will have been acquired from
	// some other API call (see Example_getTokenViaHTTP)
	token, err := createToken("foo")
	fatal(err)

	// Make request.  See func restrictedHandler for example request processor
	req, err := http.NewRequest("GET", fmt.Sprintf("http://localhost:%v/restricted", serverPort), nil)
	fatal(err)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %v", token))
	res, err := http.DefaultClient.Do(req)
	fatal(err)

	// Read the response body
	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()
	fmt.Println(buf.String())

	// Output: Welcome, foo
}

func createToken(user string) (string, error) {
	// create a signer for rsa 256
	t := jwt.New(jwt.GetSigningMethod("RS256"))

	// set our claims
	t.Claims = &CustomClaimsExample{
		&jwt.StandardClaims{
			// set the expire time
			// see http://tools.ietf.org/html/draft-ietf-oauth-json-web-token-20#section-4.1.4
			ExpiresAt: time.Now().Add(time.Minute * 1).Unix(),
		},
		"level1",
		CustomerInfo{user, "human"},
	}

	// Creat token string
	return t.SignedString(signKey)
}

// reads the form values, checks them and creates the token
func authHandler(w http.ResponseWriter, r *http.Request) {
	// make sure its post
	if r.Method != "POST" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, "No POST", r.Method)
		return
	}

	user := r.FormValue("user")
	pass := r.FormValue("pass")

	log.Printf("Authenticate: user[%s] pass[%s]\n", user, pass)

	// check values
	if user != "test" || pass != "known" {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintln(w, "Wrong info")
		return
	}

	tokenString, err := createToken(user)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "Sorry, error while Signing Token!")
		log.Printf("Token Signing error: %v\n", err)
		return
	}

	w.Header().Set("Content-Type", "application/jwt")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, tokenString)
}

// only accessible with a valid token
func restrictedHandler(w http.ResponseWriter, r *http.Request) {
	// Get token from request
	token, err := request.ParseFromRequestWithClaims(r, request.OAuth2Extractor, &CustomClaimsExample{}, func(token *jwt.Token) (interface{}, error) {
		// since we only use the one private key to sign the tokens,
		// we also only use its public counter part to verify
		return verifyKey, nil
	})

	// If the token is missing or invalid, return error
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprintln(w, "Invalid token:", err)
		return
	}

	// Token is valid
	fmt.Fprintln(w, "Welcome,", token.Claims.(*CustomClaimsExample).Name)
	return
}
