package gossip

import (
	ct "github.com/google/certificate-transparency/go"
)

// STHVersion reflects the STH Version field in RFC6862[-bis]
type STHVersion int

// STHVersion constants
const (
	STHVersion0 = 0
	STHVersion1 = 1
)

// SCTFeedbackEntry represents a single piece of SCT feedback.
type SCTFeedbackEntry struct {
	X509Chain []string `json:"x509_chain"`
	SCTData   []string `json:"sct_data"`
}

// SCTFeedback represents a collection of SCTFeedback which a client might send together.
type SCTFeedback struct {
	Feedback []SCTFeedbackEntry `json:"sct_feedback"`
}

// STHPollination represents a collection of STH pollination entries which a client might send together.
type STHPollination struct {
	STHs []ct.SignedTreeHead `json:"sths"`
}
