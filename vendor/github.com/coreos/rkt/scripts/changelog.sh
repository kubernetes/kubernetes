#!/usr/bin/env bash
#
# Create a changelog.
#
# The env variable RANGE specifies the range of commits to be searched for the changelog.
# If unset the latest tag until origin/master will be set.
#
# The env variable GITHUB_AUTH can be set in the form user:token to specify a GitHub
# personal access token. Otherwise one could run into GitHub rate limits.
# Go to https://github.com/settings/tokens to generate a token.
#
set -e

jq --version >/dev/null 2>&1 || {
    echo "could not find jq (JSON command line processor), is it installed?"
    exit 255
}

if [ -z "${RANGE}" ]; then
    LATEST_TAG=$(git describe --abbrev=0)
    RANGE="${LATEST_TAG}..origin/master"
fi

if [ ! -z "${GITHUB_AUTH}" ]; then
    GITHUB_AUTH="-u ${GITHUB_AUTH}"
fi

for pr in $(git log --pretty=%s --first-parent "${RANGE}" | egrep -o '#\w+' | tr -d '#'); do
    body=$(curl -s "${GITHUB_AUTH}" https://api.github.com/repos/coreos/rkt/pulls/"${pr}" | \
                  jq -r '{title: .title, body: .body}')

    echo "-" \
         "$(echo "${body}" | jq -r .title | sed 's/\.$//g')" \
         "([#${pr}](https://github.com/coreos/rkt/pull/$pr))." \
         "$(echo "${body}" | jq -r .body | awk -v RS='\r\n\r\n' NR==1 | tr -d '\r')"
done
