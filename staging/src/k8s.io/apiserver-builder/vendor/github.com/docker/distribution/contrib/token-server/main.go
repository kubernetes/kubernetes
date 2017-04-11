package main

import (
	"encoding/json"
	"flag"
	"net/http"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/auth"
	_ "github.com/docker/distribution/registry/auth/htpasswd"
	"github.com/docker/libtrust"
	"github.com/gorilla/mux"
)

func main() {
	var (
		issuer = &TokenIssuer{}
		pkFile string
		addr   string
		debug  bool
		err    error

		passwdFile string
		realm      string

		cert    string
		certKey string
	)

	flag.StringVar(&issuer.Issuer, "issuer", "distribution-token-server", "Issuer string for token")
	flag.StringVar(&pkFile, "key", "", "Private key file")
	flag.StringVar(&addr, "addr", "localhost:8080", "Address to listen on")
	flag.BoolVar(&debug, "debug", false, "Debug mode")

	flag.StringVar(&passwdFile, "passwd", ".htpasswd", "Passwd file")
	flag.StringVar(&realm, "realm", "", "Authentication realm")

	flag.StringVar(&cert, "tlscert", "", "Certificate file for TLS")
	flag.StringVar(&certKey, "tlskey", "", "Certificate key for TLS")

	flag.Parse()

	if debug {
		logrus.SetLevel(logrus.DebugLevel)
	}

	if pkFile == "" {
		issuer.SigningKey, err = libtrust.GenerateECP256PrivateKey()
		if err != nil {
			logrus.Fatalf("Error generating private key: %v", err)
		}
		logrus.Debugf("Using newly generated key with id %s", issuer.SigningKey.KeyID())
	} else {
		issuer.SigningKey, err = libtrust.LoadKeyFile(pkFile)
		if err != nil {
			logrus.Fatalf("Error loading key file %s: %v", pkFile, err)
		}
		logrus.Debugf("Loaded private key with id %s", issuer.SigningKey.KeyID())
	}

	if realm == "" {
		logrus.Fatalf("Must provide realm")
	}

	ac, err := auth.GetAccessController("htpasswd", map[string]interface{}{
		"realm": realm,
		"path":  passwdFile,
	})
	if err != nil {
		logrus.Fatalf("Error initializing access controller: %v", err)
	}

	ctx := context.Background()

	ts := &tokenServer{
		issuer:           issuer,
		accessController: ac,
	}

	router := mux.NewRouter()
	router.Path("/token/").Methods("GET").Handler(handlerWithContext(ctx, ts.getToken))

	if cert == "" {
		err = http.ListenAndServe(addr, router)
	} else if certKey == "" {
		logrus.Fatalf("Must provide certficate (-tlscert) and key (-tlskey)")
	} else {
		err = http.ListenAndServeTLS(addr, cert, certKey, router)
	}

	if err != nil {
		logrus.Infof("Error serving: %v", err)
	}

}

// handlerWithContext wraps the given context-aware handler by setting up the
// request context from a base context.
func handlerWithContext(ctx context.Context, handler func(context.Context, http.ResponseWriter, *http.Request)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := context.WithRequest(ctx, r)
		logger := context.GetRequestLogger(ctx)
		ctx = context.WithLogger(ctx, logger)

		handler(ctx, w, r)
	})
}

func handleError(ctx context.Context, err error, w http.ResponseWriter) {
	ctx, w = context.WithResponseWriter(ctx, w)

	if serveErr := errcode.ServeJSON(w, err); serveErr != nil {
		context.GetResponseLogger(ctx).Errorf("error sending error response: %v", serveErr)
		return
	}

	context.GetResponseLogger(ctx).Info("application error")
}

type tokenServer struct {
	issuer           *TokenIssuer
	accessController auth.AccessController
}

// getToken handles authenticating the request and authorizing access to the
// requested scopes.
func (ts *tokenServer) getToken(ctx context.Context, w http.ResponseWriter, r *http.Request) {
	context.GetLogger(ctx).Info("getToken")

	params := r.URL.Query()
	service := params.Get("service")
	scopeSpecifiers := params["scope"]

	requestedAccessList := ResolveScopeSpecifiers(ctx, scopeSpecifiers)

	authorizedCtx, err := ts.accessController.Authorized(ctx, requestedAccessList...)
	if err != nil {
		challenge, ok := err.(auth.Challenge)
		if !ok {
			handleError(ctx, err, w)
			return
		}

		// Get response context.
		ctx, w = context.WithResponseWriter(ctx, w)

		challenge.SetHeaders(w)
		handleError(ctx, errcode.ErrorCodeUnauthorized.WithDetail(challenge.Error()), w)

		context.GetResponseLogger(ctx).Info("get token authentication challenge")

		return
	}
	ctx = authorizedCtx

	username := context.GetStringValue(ctx, "auth.user.name")

	ctx = context.WithValue(ctx, "acctSubject", username)
	ctx = context.WithLogger(ctx, context.GetLogger(ctx, "acctSubject"))

	context.GetLogger(ctx).Info("authenticated client")

	ctx = context.WithValue(ctx, "requestedAccess", requestedAccessList)
	ctx = context.WithLogger(ctx, context.GetLogger(ctx, "requestedAccess"))

	scopePrefix := username + "/"
	grantedAccessList := make([]auth.Access, 0, len(requestedAccessList))
	for _, access := range requestedAccessList {
		if access.Type != "repository" {
			context.GetLogger(ctx).Debugf("Skipping unsupported resource type: %s", access.Type)
			continue
		}
		if !strings.HasPrefix(access.Name, scopePrefix) {
			context.GetLogger(ctx).Debugf("Resource scope not allowed: %s", access.Name)
			continue
		}
		grantedAccessList = append(grantedAccessList, access)
	}

	ctx = context.WithValue(ctx, "grantedAccess", grantedAccessList)
	ctx = context.WithLogger(ctx, context.GetLogger(ctx, "grantedAccess"))

	token, err := ts.issuer.CreateJWT(username, service, grantedAccessList)
	if err != nil {
		handleError(ctx, err, w)
		return
	}

	context.GetLogger(ctx).Info("authorized client")

	// Get response context.
	ctx, w = context.WithResponseWriter(ctx, w)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"token": token})

	context.GetResponseLogger(ctx).Info("get token complete")
}
