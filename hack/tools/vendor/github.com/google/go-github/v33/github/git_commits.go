// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"golang.org/x/crypto/openpgp"
)

// SignatureVerification represents GPG signature verification.
type SignatureVerification struct {
	Verified  *bool   `json:"verified,omitempty"`
	Reason    *string `json:"reason,omitempty"`
	Signature *string `json:"signature,omitempty"`
	Payload   *string `json:"payload,omitempty"`
}

// Commit represents a GitHub commit.
type Commit struct {
	SHA          *string                `json:"sha,omitempty"`
	Author       *CommitAuthor          `json:"author,omitempty"`
	Committer    *CommitAuthor          `json:"committer,omitempty"`
	Message      *string                `json:"message,omitempty"`
	Tree         *Tree                  `json:"tree,omitempty"`
	Parents      []*Commit              `json:"parents,omitempty"`
	Stats        *CommitStats           `json:"stats,omitempty"`
	HTMLURL      *string                `json:"html_url,omitempty"`
	URL          *string                `json:"url,omitempty"`
	Verification *SignatureVerification `json:"verification,omitempty"`
	NodeID       *string                `json:"node_id,omitempty"`

	// CommentCount is the number of GitHub comments on the commit. This
	// is only populated for requests that fetch GitHub data like
	// Pulls.ListCommits, Repositories.ListCommits, etc.
	CommentCount *int `json:"comment_count,omitempty"`

	// SigningKey denotes a key to sign the commit with. If not nil this key will
	// be used to sign the commit. The private key must be present and already
	// decrypted. Ignored if Verification.Signature is defined.
	SigningKey *openpgp.Entity `json:"-"`
}

func (c Commit) String() string {
	return Stringify(c)
}

// CommitAuthor represents the author or committer of a commit. The commit
// author may not correspond to a GitHub User.
type CommitAuthor struct {
	Date  *time.Time `json:"date,omitempty"`
	Name  *string    `json:"name,omitempty"`
	Email *string    `json:"email,omitempty"`

	// The following fields are only populated by Webhook events.
	Login *string `json:"username,omitempty"` // Renamed for go-github consistency.
}

func (c CommitAuthor) String() string {
	return Stringify(c)
}

// GetCommit fetches the Commit object for a given SHA.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#get-a-commit
func (s *GitService) GetCommit(ctx context.Context, owner string, repo string, sha string) (*Commit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/commits/%v", owner, repo, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// createCommit represents the body of a CreateCommit request.
type createCommit struct {
	Author    *CommitAuthor `json:"author,omitempty"`
	Committer *CommitAuthor `json:"committer,omitempty"`
	Message   *string       `json:"message,omitempty"`
	Tree      *string       `json:"tree,omitempty"`
	Parents   []string      `json:"parents,omitempty"`
	Signature *string       `json:"signature,omitempty"`
}

// CreateCommit creates a new commit in a repository.
// commit must not be nil.
//
// The commit.Committer is optional and will be filled with the commit.Author
// data if omitted. If the commit.Author is omitted, it will be filled in with
// the authenticated user’s information and the current date.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#create-a-commit
func (s *GitService) CreateCommit(ctx context.Context, owner string, repo string, commit *Commit) (*Commit, *Response, error) {
	if commit == nil {
		return nil, nil, fmt.Errorf("commit must be provided")
	}

	u := fmt.Sprintf("repos/%v/%v/git/commits", owner, repo)

	parents := make([]string, len(commit.Parents))
	for i, parent := range commit.Parents {
		parents[i] = *parent.SHA
	}

	body := &createCommit{
		Author:    commit.Author,
		Committer: commit.Committer,
		Message:   commit.Message,
		Parents:   parents,
	}
	if commit.Tree != nil {
		body.Tree = commit.Tree.SHA
	}
	if commit.SigningKey != nil {
		signature, err := createSignature(commit.SigningKey, body)
		if err != nil {
			return nil, nil, err
		}
		body.Signature = &signature
	}
	if commit.Verification != nil {
		body.Signature = commit.Verification.Signature
	}

	req, err := s.client.NewRequest("POST", u, body)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

func createSignature(signingKey *openpgp.Entity, commit *createCommit) (string, error) {
	if signingKey == nil || commit == nil {
		return "", errors.New("createSignature: invalid parameters")
	}

	message, err := createSignatureMessage(commit)
	if err != nil {
		return "", err
	}

	writer := new(bytes.Buffer)
	reader := bytes.NewReader([]byte(message))
	if err := openpgp.ArmoredDetachSign(writer, signingKey, reader, nil); err != nil {
		return "", err
	}

	return writer.String(), nil
}

func createSignatureMessage(commit *createCommit) (string, error) {
	if commit == nil || commit.Message == nil || *commit.Message == "" || commit.Author == nil {
		return "", errors.New("createSignatureMessage: invalid parameters")
	}

	var message []string

	if commit.Tree != nil {
		message = append(message, fmt.Sprintf("tree %s", *commit.Tree))
	}

	for _, parent := range commit.Parents {
		message = append(message, fmt.Sprintf("parent %s", parent))
	}

	message = append(message, fmt.Sprintf("author %s <%s> %d %s", commit.Author.GetName(), commit.Author.GetEmail(), commit.Author.GetDate().Unix(), commit.Author.GetDate().Format("-0700")))

	committer := commit.Committer
	if committer == nil {
		committer = commit.Author
	}

	// There needs to be a double newline after committer
	message = append(message, fmt.Sprintf("committer %s <%s> %d %s\n", committer.GetName(), committer.GetEmail(), committer.GetDate().Unix(), committer.GetDate().Format("-0700")))
	message = append(message, *commit.Message)

	return strings.Join(message, "\n"), nil
}
