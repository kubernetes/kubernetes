//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package middleware

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"github.com/auth0/go-jwt-middleware"
	jwt "github.com/dgrijalva/jwt-go"
	"github.com/gorilla/context"
	"net/http"
)

var (
	required_claims = []string{"iss", "iat", "exp"}
)

type JwtAuth struct {
	adminKey []byte
	userKey  []byte
}

type Issuer struct {
	PrivateKey string `json:"key"`
}

type JwtAuthConfig struct {
	Admin Issuer `json:"admin"`
	User  Issuer `json:"user"`
}

func generate_qsh(r *http.Request) string {
	// Please see Heketi REST API for more information
	claim := r.Method + "&" + r.URL.Path
	hash := sha256.New()
	hash.Write([]byte(claim))
	return hex.EncodeToString(hash.Sum(nil))
}

func NewJwtAuth(config *JwtAuthConfig) *JwtAuth {

	if config.Admin.PrivateKey == "" ||
		config.User.PrivateKey == "" {
		return nil
	}

	j := &JwtAuth{}
	j.adminKey = []byte(config.Admin.PrivateKey)
	j.userKey = []byte(config.User.PrivateKey)

	return j
}

func (j *JwtAuth) ServeHTTP(w http.ResponseWriter, r *http.Request, next http.HandlerFunc) {

	// Access token from header
	rawtoken, err := jwtmiddleware.FromAuthHeader(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Determine if we have the token
	if rawtoken == "" {
		http.Error(w, "Required authorization token not found", http.StatusUnauthorized)
		return
	}

	// Parse token
	var claims jwt.MapClaims
	token, err := jwt.Parse(rawtoken, func(token *jwt.Token) (interface{}, error) {

		// Verify Method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("Unexpected signing method: %v", token.Header["alg"])
		}

		claims = token.Claims.(jwt.MapClaims)
		if claims == nil {
			return nil, fmt.Errorf("No claims found in token")
		}

		// Get claims
		if issuer, ok := claims["iss"]; ok {
			switch issuer {
			case "admin":
				return j.adminKey, nil
			case "user":
				return j.userKey, nil
			default:
				return nil, errors.New("Unknown user")
			}
		}

		return nil, errors.New("Token missing iss claim")
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	if !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	// Check for required claims
	for _, required_claim := range required_claims {
		if _, ok := claims[required_claim]; !ok {
			// Claim missing
			http.Error(w, fmt.Sprintf("Required claim %v missing from token", required_claim), http.StatusBadRequest)
			return
		}
	}

	// Check qsh claim
	if claims["qsh"] != generate_qsh(r) {
		http.Error(w, "Invalid qsh claim in token", http.StatusUnauthorized)
		return
	}

	// Store token in request for other middleware to access
	context.Set(r, "jwt", token)

	// Everything passes call next middleware
	next(w, r)
}
