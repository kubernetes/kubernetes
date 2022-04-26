/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package internal

import (
	"time"

	"github.com/google/go-github/v33/github"
	"github.com/sirupsen/logrus"
)

const (
	// MaxGithubRetries is the maximum amount of times we flag a GitHub error as
	// retryable before we give up and do not flag the same call as retryable
	// anymore.
	MaxGithubRetries = 3

	// defaultGithubSleep is the amount of time we wait between two consecutive
	// GitHub calls in case we cannot extract that information from the error
	// itself.
	defaultGithubSleep = time.Minute
)

// DefaultGithubErrChecker is a GithubErrChecker set up with a default amount
// of retries and the default sleep function.
func DefaultGithubErrChecker() func(error) bool {
	return GithubErrChecker(MaxGithubRetries, time.Sleep)
}

// GithubErrChecker returns a function that checks errors from GitHub and
// decides if they can / should be retried.
// It needs to be called with `maxTries`, a number of retries a single call
// should be retried at max, and `sleeper`, a function which implements the
// sleeping.
//
// Currently the only special error that is flagged as retryable is the
// `AbuseRateLimitError`. If such an error occurs, we sleep for a while (the
// amount of time the error told us to wait) and then report back that we can
// retry.
// Other special errors should be easy to implement too.
//
// It can be used like this:
//  for shouldRetry := GithubErrChecker(10, time.Sleep); ; {
//    commit, res, err := github_client.GetCommit(...)
//    if !shouldRetry(err) {
//      return commit, res, err
//    }
//  }
func GithubErrChecker(maxTries int, sleeper func(time.Duration)) func(error) bool {
	try := 0

	return func(err error) bool {
		if err == nil {
			return false
		}
		if try >= maxTries {
			logrus.Errorf("Max retries (%d) reached, not retrying anymore: %v", maxTries, err)
			return false
		}

		try++

		if aerr, ok := err.(*github.AbuseRateLimitError); ok {
			waitDuration := defaultGithubSleep
			if d := aerr.RetryAfter; d != nil {
				waitDuration = *d
			}
			logrus.
				WithField("err", aerr).
				Infof("Hit the abuse rate limit on try %d, sleeping for %s", try, waitDuration)
			sleeper(waitDuration)
			return true
		}

		return false
	}
}
