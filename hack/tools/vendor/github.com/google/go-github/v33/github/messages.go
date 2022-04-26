// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides functions for validating payloads from GitHub Webhooks.
// GitHub API docs: https://developer.github.com/webhooks/securing/#validating-payloads-from-github

package github

import (
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
)

const (
	// sha1Prefix is the prefix used by GitHub before the HMAC hexdigest.
	sha1Prefix = "sha1"
	// sha256Prefix and sha512Prefix are provided for future compatibility.
	sha256Prefix = "sha256"
	sha512Prefix = "sha512"
	// signatureHeader is the GitHub header key used to pass the HMAC hexdigest.
	signatureHeader = "X-Hub-Signature"
	// eventTypeHeader is the GitHub header key used to pass the event type.
	eventTypeHeader = "X-Github-Event"
	// deliveryIDHeader is the GitHub header key used to pass the unique ID for the webhook event.
	deliveryIDHeader = "X-Github-Delivery"
)

var (
	// eventTypeMapping maps webhooks types to their corresponding go-github struct types.
	eventTypeMapping = map[string]string{
		"check_run":                      "CheckRunEvent",
		"check_suite":                    "CheckSuiteEvent",
		"commit_comment":                 "CommitCommentEvent",
		"content_reference":              "ContentReferenceEvent",
		"create":                         "CreateEvent",
		"delete":                         "DeleteEvent",
		"deploy_key":                     "DeployKeyEvent",
		"deployment":                     "DeploymentEvent",
		"deployment_status":              "DeploymentStatusEvent",
		"fork":                           "ForkEvent",
		"github_app_authorization":       "GitHubAppAuthorizationEvent",
		"gollum":                         "GollumEvent",
		"installation":                   "InstallationEvent",
		"installation_repositories":      "InstallationRepositoriesEvent",
		"issue_comment":                  "IssueCommentEvent",
		"issues":                         "IssuesEvent",
		"label":                          "LabelEvent",
		"marketplace_purchase":           "MarketplacePurchaseEvent",
		"member":                         "MemberEvent",
		"membership":                     "MembershipEvent",
		"meta":                           "MetaEvent",
		"milestone":                      "MilestoneEvent",
		"organization":                   "OrganizationEvent",
		"org_block":                      "OrgBlockEvent",
		"package":                        "PackageEvent",
		"page_build":                     "PageBuildEvent",
		"ping":                           "PingEvent",
		"project":                        "ProjectEvent",
		"project_card":                   "ProjectCardEvent",
		"project_column":                 "ProjectColumnEvent",
		"public":                         "PublicEvent",
		"pull_request_review":            "PullRequestReviewEvent",
		"pull_request_review_comment":    "PullRequestReviewCommentEvent",
		"pull_request":                   "PullRequestEvent",
		"push":                           "PushEvent",
		"repository":                     "RepositoryEvent",
		"repository_dispatch":            "RepositoryDispatchEvent",
		"repository_vulnerability_alert": "RepositoryVulnerabilityAlertEvent",
		"release":                        "ReleaseEvent",
		"star":                           "StarEvent",
		"status":                         "StatusEvent",
		"team":                           "TeamEvent",
		"team_add":                       "TeamAddEvent",
		"user":                           "UserEvent",
		"watch":                          "WatchEvent",
		"workflow_dispatch":              "WorkflowDispatchEvent",
		"workflow_run":                   "WorkflowRunEvent",
	}
)

// genMAC generates the HMAC signature for a message provided the secret key
// and hashFunc.
func genMAC(message, key []byte, hashFunc func() hash.Hash) []byte {
	mac := hmac.New(hashFunc, key)
	mac.Write(message)
	return mac.Sum(nil)
}

// checkMAC reports whether messageMAC is a valid HMAC tag for message.
func checkMAC(message, messageMAC, key []byte, hashFunc func() hash.Hash) bool {
	expectedMAC := genMAC(message, key, hashFunc)
	return hmac.Equal(messageMAC, expectedMAC)
}

// messageMAC returns the hex-decoded HMAC tag from the signature and its
// corresponding hash function.
func messageMAC(signature string) ([]byte, func() hash.Hash, error) {
	if signature == "" {
		return nil, nil, errors.New("missing signature")
	}
	sigParts := strings.SplitN(signature, "=", 2)
	if len(sigParts) != 2 {
		return nil, nil, fmt.Errorf("error parsing signature %q", signature)
	}

	var hashFunc func() hash.Hash
	switch sigParts[0] {
	case sha1Prefix:
		hashFunc = sha1.New
	case sha256Prefix:
		hashFunc = sha256.New
	case sha512Prefix:
		hashFunc = sha512.New
	default:
		return nil, nil, fmt.Errorf("unknown hash type prefix: %q", sigParts[0])
	}

	buf, err := hex.DecodeString(sigParts[1])
	if err != nil {
		return nil, nil, fmt.Errorf("error decoding signature %q: %v", signature, err)
	}
	return buf, hashFunc, nil
}

// ValidatePayload validates an incoming GitHub Webhook event request
// and returns the (JSON) payload.
// The Content-Type header of the payload can be "application/json" or "application/x-www-form-urlencoded".
// If the Content-Type is neither then an error is returned.
// secretToken is the GitHub Webhook secret token.
// If your webhook does not contain a secret token, you can pass nil or an empty slice.
// This is intended for local development purposes only and all webhooks should ideally set up a secret token.
//
// Example usage:
//
//     func (s *GitHubEventMonitor) ServeHTTP(w http.ResponseWriter, r *http.Request) {
//       payload, err := github.ValidatePayload(r, s.webhookSecretKey)
//       if err != nil { ... }
//       // Process payload...
//     }
//
func ValidatePayload(r *http.Request, secretToken []byte) (payload []byte, err error) {
	var body []byte // Raw body that GitHub uses to calculate the signature.

	switch ct := r.Header.Get("Content-Type"); ct {
	case "application/json":
		var err error
		if body, err = ioutil.ReadAll(r.Body); err != nil {
			return nil, err
		}

		// If the content type is application/json,
		// the JSON payload is just the original body.
		payload = body

	case "application/x-www-form-urlencoded":
		// payloadFormParam is the name of the form parameter that the JSON payload
		// will be in if a webhook has its content type set to application/x-www-form-urlencoded.
		const payloadFormParam = "payload"

		var err error
		if body, err = ioutil.ReadAll(r.Body); err != nil {
			return nil, err
		}

		// If the content type is application/x-www-form-urlencoded,
		// the JSON payload will be under the "payload" form param.
		form, err := url.ParseQuery(string(body))
		if err != nil {
			return nil, err
		}
		payload = []byte(form.Get(payloadFormParam))

	default:
		return nil, fmt.Errorf("Webhook request has unsupported Content-Type %q", ct)
	}

	// Only validate the signature if a secret token exists. This is intended for
	// local development only and all webhooks should ideally set up a secret token.
	if len(secretToken) > 0 {
		sig := r.Header.Get(signatureHeader)
		if err := ValidateSignature(sig, body, secretToken); err != nil {
			return nil, err
		}
	}

	return payload, nil
}

// ValidateSignature validates the signature for the given payload.
// signature is the GitHub hash signature delivered in the X-Hub-Signature header.
// payload is the JSON payload sent by GitHub Webhooks.
// secretToken is the GitHub Webhook secret token.
//
// GitHub API docs: https://developer.github.com/webhooks/securing/#validating-payloads-from-github
func ValidateSignature(signature string, payload, secretToken []byte) error {
	messageMAC, hashFunc, err := messageMAC(signature)
	if err != nil {
		return err
	}
	if !checkMAC(payload, messageMAC, secretToken, hashFunc) {
		return errors.New("payload signature check failed")
	}
	return nil
}

// WebHookType returns the event type of webhook request r.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/hooks/#webhook-headers
func WebHookType(r *http.Request) string {
	return r.Header.Get(eventTypeHeader)
}

// DeliveryID returns the unique delivery ID of webhook request r.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/hooks/#webhook-headers
func DeliveryID(r *http.Request) string {
	return r.Header.Get(deliveryIDHeader)
}

// ParseWebHook parses the event payload. For recognized event types, a
// value of the corresponding struct type will be returned (as returned
// by Event.ParsePayload()). An error will be returned for unrecognized event
// types.
//
// Example usage:
//
//     func (s *GitHubEventMonitor) ServeHTTP(w http.ResponseWriter, r *http.Request) {
//       payload, err := github.ValidatePayload(r, s.webhookSecretKey)
//       if err != nil { ... }
//       event, err := github.ParseWebHook(github.WebHookType(r), payload)
//       if err != nil { ... }
//       switch event := event.(type) {
//       case *github.CommitCommentEvent:
//           processCommitCommentEvent(event)
//       case *github.CreateEvent:
//           processCreateEvent(event)
//       ...
//       }
//     }
//
func ParseWebHook(messageType string, payload []byte) (interface{}, error) {
	eventType, ok := eventTypeMapping[messageType]
	if !ok {
		return nil, fmt.Errorf("unknown X-Github-Event in message: %v", messageType)
	}

	event := Event{
		Type:       &eventType,
		RawPayload: (*json.RawMessage)(&payload),
	}
	return event.ParsePayload()
}
