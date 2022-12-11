// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

var serviceAccountImpersonationRE = regexp.MustCompile("https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/(.*@.*):generateAccessToken")

const (
	executableSupportedMaxVersion = 1
	defaultTimeout                = 30 * time.Second
	timeoutMinimum                = 5 * time.Second
	timeoutMaximum                = 120 * time.Second
	executableSource              = "response"
	outputFileSource              = "output file"
)

type nonCacheableError struct {
	message string
}

func (nce nonCacheableError) Error() string {
	return nce.message
}

func missingFieldError(source, field string) error {
	return fmt.Errorf("oauth2/google: %v missing `%q` field", source, field)
}

func jsonParsingError(source, data string) error {
	return fmt.Errorf("oauth2/google: unable to parse %v\nResponse: %v", source, data)
}

func malformedFailureError() error {
	return nonCacheableError{"oauth2/google: response must include `error` and `message` fields when unsuccessful"}
}

func userDefinedError(code, message string) error {
	return nonCacheableError{fmt.Sprintf("oauth2/google: response contains unsuccessful response: (%v) %v", code, message)}
}

func unsupportedVersionError(source string, version int) error {
	return fmt.Errorf("oauth2/google: %v contains unsupported version: %v", source, version)
}

func tokenExpiredError() error {
	return nonCacheableError{"oauth2/google: the token returned by the executable is expired"}
}

func tokenTypeError(source string) error {
	return fmt.Errorf("oauth2/google: %v contains unsupported token type", source)
}

func exitCodeError(exitCode int) error {
	return fmt.Errorf("oauth2/google: executable command failed with exit code %v", exitCode)
}

func executableError(err error) error {
	return fmt.Errorf("oauth2/google: executable command failed: %v", err)
}

func executablesDisallowedError() error {
	return errors.New("oauth2/google: executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run")
}

func timeoutRangeError() error {
	return errors.New("oauth2/google: invalid `timeout_millis` field — executable timeout must be between 5 and 120 seconds")
}

func commandMissingError() error {
	return errors.New("oauth2/google: missing `command` field — executable command must be provided")
}

type environment interface {
	existingEnv() []string
	getenv(string) string
	run(ctx context.Context, command string, env []string) ([]byte, error)
	now() time.Time
}

type runtimeEnvironment struct{}

func (r runtimeEnvironment) existingEnv() []string {
	return os.Environ()
}

func (r runtimeEnvironment) getenv(key string) string {
	return os.Getenv(key)
}

func (r runtimeEnvironment) now() time.Time {
	return time.Now().UTC()
}

func (r runtimeEnvironment) run(ctx context.Context, command string, env []string) ([]byte, error) {
	splitCommand := strings.Fields(command)
	cmd := exec.CommandContext(ctx, splitCommand[0], splitCommand[1:]...)
	cmd.Env = env

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, context.DeadlineExceeded
		}

		if exitError, ok := err.(*exec.ExitError); ok {
			return nil, exitCodeError(exitError.ExitCode())
		}

		return nil, executableError(err)
	}

	bytesStdout := bytes.TrimSpace(stdout.Bytes())
	if len(bytesStdout) > 0 {
		return bytesStdout, nil
	}
	return bytes.TrimSpace(stderr.Bytes()), nil
}

type executableCredentialSource struct {
	Command    string
	Timeout    time.Duration
	OutputFile string
	ctx        context.Context
	config     *Config
	env        environment
}

// CreateExecutableCredential creates an executableCredentialSource given an ExecutableConfig.
// It also performs defaulting and type conversions.
func CreateExecutableCredential(ctx context.Context, ec *ExecutableConfig, config *Config) (executableCredentialSource, error) {
	if ec.Command == "" {
		return executableCredentialSource{}, commandMissingError()
	}

	result := executableCredentialSource{}
	result.Command = ec.Command
	if ec.TimeoutMillis == nil {
		result.Timeout = defaultTimeout
	} else {
		result.Timeout = time.Duration(*ec.TimeoutMillis) * time.Millisecond
		if result.Timeout < timeoutMinimum || result.Timeout > timeoutMaximum {
			return executableCredentialSource{}, timeoutRangeError()
		}
	}
	result.OutputFile = ec.OutputFile
	result.ctx = ctx
	result.config = config
	result.env = runtimeEnvironment{}
	return result, nil
}

type executableResponse struct {
	Version        int    `json:"version,omitempty"`
	Success        *bool  `json:"success,omitempty"`
	TokenType      string `json:"token_type,omitempty"`
	ExpirationTime int64  `json:"expiration_time,omitempty"`
	IdToken        string `json:"id_token,omitempty"`
	SamlResponse   string `json:"saml_response,omitempty"`
	Code           string `json:"code,omitempty"`
	Message        string `json:"message,omitempty"`
}

func parseSubjectTokenFromSource(response []byte, source string, now int64) (string, error) {
	var result executableResponse
	if err := json.Unmarshal(response, &result); err != nil {
		return "", jsonParsingError(source, string(response))
	}

	if result.Version == 0 {
		return "", missingFieldError(source, "version")
	}

	if result.Success == nil {
		return "", missingFieldError(source, "success")
	}

	if !*result.Success {
		if result.Code == "" || result.Message == "" {
			return "", malformedFailureError()
		}
		return "", userDefinedError(result.Code, result.Message)
	}

	if result.Version > executableSupportedMaxVersion || result.Version < 0 {
		return "", unsupportedVersionError(source, result.Version)
	}

	if result.ExpirationTime == 0 {
		return "", missingFieldError(source, "expiration_time")
	}

	if result.TokenType == "" {
		return "", missingFieldError(source, "token_type")
	}

	if result.ExpirationTime < now {
		return "", tokenExpiredError()
	}

	if result.TokenType == "urn:ietf:params:oauth:token-type:jwt" || result.TokenType == "urn:ietf:params:oauth:token-type:id_token" {
		if result.IdToken == "" {
			return "", missingFieldError(source, "id_token")
		}
		return result.IdToken, nil
	}

	if result.TokenType == "urn:ietf:params:oauth:token-type:saml2" {
		if result.SamlResponse == "" {
			return "", missingFieldError(source, "saml_response")
		}
		return result.SamlResponse, nil
	}

	return "", tokenTypeError(source)
}

func (cs executableCredentialSource) subjectToken() (string, error) {
	if token, err := cs.getTokenFromOutputFile(); token != "" || err != nil {
		return token, err
	}

	return cs.getTokenFromExecutableCommand()
}

func (cs executableCredentialSource) getTokenFromOutputFile() (token string, err error) {
	if cs.OutputFile == "" {
		// This ExecutableCredentialSource doesn't use an OutputFile.
		return "", nil
	}

	file, err := os.Open(cs.OutputFile)
	if err != nil {
		// No OutputFile found. Hasn't been created yet, so skip it.
		return "", nil
	}
	defer file.Close()

	data, err := io.ReadAll(io.LimitReader(file, 1<<20))
	if err != nil || len(data) == 0 {
		// Cachefile exists, but no data found. Get new credential.
		return "", nil
	}

	token, err = parseSubjectTokenFromSource(data, outputFileSource, cs.env.now().Unix())
	if err != nil {
		if _, ok := err.(nonCacheableError); ok {
			// If the cached token is expired we need a new token,
			// and if the cache contains a failure, we need to try again.
			return "", nil
		}

		// There was an error in the cached token, and the developer should be aware of it.
		return "", err
	}
	// Token parsing succeeded.  Use found token.
	return token, nil
}

func (cs executableCredentialSource) executableEnvironment() []string {
	result := cs.env.existingEnv()
	result = append(result, fmt.Sprintf("GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE=%v", cs.config.Audience))
	result = append(result, fmt.Sprintf("GOOGLE_EXTERNAL_ACCOUNT_TOKEN_TYPE=%v", cs.config.SubjectTokenType))
	result = append(result, "GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE=0")
	if cs.config.ServiceAccountImpersonationURL != "" {
		matches := serviceAccountImpersonationRE.FindStringSubmatch(cs.config.ServiceAccountImpersonationURL)
		if matches != nil {
			result = append(result, fmt.Sprintf("GOOGLE_EXTERNAL_ACCOUNT_IMPERSONATED_EMAIL=%v", matches[1]))
		}
	}
	if cs.OutputFile != "" {
		result = append(result, fmt.Sprintf("GOOGLE_EXTERNAL_ACCOUNT_OUTPUT_FILE=%v", cs.OutputFile))
	}
	return result
}

func (cs executableCredentialSource) getTokenFromExecutableCommand() (string, error) {
	// For security reasons, we need our consumers to set this environment variable to allow executables to be run.
	if cs.env.getenv("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES") != "1" {
		return "", executablesDisallowedError()
	}

	ctx, cancel := context.WithDeadline(cs.ctx, cs.env.now().Add(cs.Timeout))
	defer cancel()

	output, err := cs.env.run(ctx, cs.Command, cs.executableEnvironment())
	if err != nil {
		return "", err
	}
	return parseSubjectTokenFromSource(output, executableSource, cs.env.now().Unix())
}
