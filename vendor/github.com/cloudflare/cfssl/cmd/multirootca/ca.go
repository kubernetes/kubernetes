package main

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"errors"
	"flag"
	"net"
	"net/http"

	"github.com/cloudflare/cfssl/api/info"
	"github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/multiroot/config"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
	"github.com/cloudflare/cfssl/whitelist"

	_ "github.com/lib/pq" // import to support Postgres
)

func parseSigner(root *config.Root) (signer.Signer, error) {
	privateKey := root.PrivateKey
	switch priv := privateKey.(type) {
	case *rsa.PrivateKey, *ecdsa.PrivateKey:
		s, err := local.NewSigner(priv, root.Certificate, signer.DefaultSigAlgo(priv), nil)
		if err != nil {
			return nil, err
		}
		s.SetPolicy(root.Config)
		if root.DB != nil {
			dbAccessor := sql.NewAccessor(root.DB)
			s.SetDBAccessor(dbAccessor)
		}
		return s, nil
	default:
		return nil, errors.New("unsupported private key type")
	}
}

var (
	defaultLabel string
	signers      = map[string]signer.Signer{}
	whitelists   = map[string]whitelist.NetACL{}
)

func main() {
	flagAddr := flag.String("a", ":8888", "listening address")
	flagRootFile := flag.String("roots", "", "configuration file specifying root keys")
	flagDefaultLabel := flag.String("l", "", "specify a default label")
	flagEndpointCert := flag.String("tls-cert", "", "server certificate")
	flagEndpointKey := flag.String("tls-key", "", "server private key")
	flag.Parse()

	if *flagRootFile == "" {
		log.Fatal("no root file specified")
	}

	roots, err := config.Parse(*flagRootFile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	for label, root := range roots {
		s, err := parseSigner(root)
		if err != nil {
			log.Criticalf("%v", err)
		}
		signers[label] = s
		if root.ACL != nil {
			whitelists[label] = root.ACL
		}
		log.Info("loaded signer ", label)
	}

	defaultLabel = *flagDefaultLabel
	initStats()

	infoHandler, err := info.NewMultiHandler(signers, defaultLabel)
	if err != nil {
		log.Criticalf("%v", err)
	}

	var localhost = whitelist.NewBasic()
	localhost.Add(net.ParseIP("127.0.0.1"))
	localhost.Add(net.ParseIP("::1"))
	metrics, err := whitelist.NewHandlerFunc(dumpMetrics, metricsDisallowed, localhost)
	if err != nil {
		log.Criticalf("failed to set up the metrics whitelist: %v", err)
	}

	http.HandleFunc("/api/v1/cfssl/authsign", dispatchRequest)
	http.Handle("/api/v1/cfssl/info", infoHandler)
	http.Handle("/api/v1/cfssl/metrics", metrics)

	if *flagEndpointCert == "" && *flagEndpointKey == "" {
		log.Info("Now listening on ", *flagAddr)
		log.Fatal(http.ListenAndServe(*flagAddr, nil))
	} else {

		log.Info("Now listening on https:// ", *flagAddr)
		log.Fatal(http.ListenAndServeTLS(*flagAddr, *flagEndpointCert, *flagEndpointKey, nil))
	}

}
