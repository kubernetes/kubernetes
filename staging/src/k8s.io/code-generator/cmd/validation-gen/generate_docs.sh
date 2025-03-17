#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

# Directory where the script is located (staging/src/k8s.io/code-generator/validation-gen/)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run the command and capture the JSON output
json_output=$(go run "${script_dir}" --docs)

# Create a Markdown file in the same directory
markdown_file="${script_dir}/validation_tags.md"

# Check if any tags have args
has_args=false
echo "$json_output" | jq -c '.[]' | while read -r line; do
    args=$(echo "$line" | jq -r '.Args')
    if [[ "$args" != "null" ]]; then
        has_args=true
        break
    fi
done

# Write the header to the Markdown file
{
    echo "# Kubernetes Validation Tags Documentation"
    echo ""
    echo "This document lists the supported validation tags and their related information."
    echo ""
    echo "## Tags Overview"
    echo ""
    if [[ "$has_args" == "true" ]]; then
        echo "| Tag | Usage | Args | Description | Scopes |"
        echo "|-----|-------------|------|-------------|----------|"
    else
        echo "| Tag | Usage | Description | Scopes |"
        echo "|-----|-------------|-------------|----------|"
    fi
} > "$markdown_file"

# Process the JSON output and populate the main table
{
    echo "$json_output" | jq -c '.[]' | while read -r line; do
        tag=$(echo "$line" | jq -r '.Tag')
        description=$(echo "$line" | jq -r '.Description')
        scopes=$(echo "$line" | jq -r '.Scopes | join(", ")')
        usage=$(echo "$line" | jq -r '.Usage')
        args=$(echo "$line" | jq -r '.Args')

        # Add row to main table
        tag_link=$(echo "$tag" | sed 's/k8s:/k8s/' | tr '[:upper:]' '[:lower:]')
        formatted_usage=$(echo "$usage" | sed 's/</\\</g' | sed 's/>/\\>/g')
        
        if [[ "$has_args" == "true" ]]; then
            # Format Args for the main table
            if [[ "$args" != "null" ]]; then
                args_formatted=$(echo "$args" | jq -r '.[] | .Description' | paste -sd "," -)
                # Escape < and > in args_formatted to match the usage formatting
                args_formatted=$(echo "$args_formatted" | sed 's/</\\</g' | sed 's/>/\\>/g')
            else
                args_formatted=""
            fi
            echo "| [\`$tag\`](#$tag_link) | $formatted_usage | $args_formatted | $description | $scopes |"
        else
            echo "| [\`$tag\`](#$tag_link) | $formatted_usage | $description | $scopes |"
        fi
    done

    echo ""
    echo "## Tag Details"
    echo ""
} >> "$markdown_file"

# Create separate sections for each tag's payloads and args
echo "$json_output" | jq -c '.[]' | while read -r line; do
    tag=$(echo "$line" | jq -r '.Tag')
    payloads=$(echo "$line" | jq -r '.Payloads')
    payloads_type=$(echo "$line" | jq -r '.PayloadsType')
    payloads_required=$(echo "$line" | jq -r '.PayloadsRequired')
    args=$(echo "$line" | jq -r '.Args')
    docs=$(echo "$line" | jq -r '.Docs')

    # Create section for this tag
    {
        echo "### $tag"
        echo ""

        # Add docs if available
        if [[ "$docs" != "" && "$docs" != "null" ]]; then
            echo "$docs"
            echo ""
        fi

        # Add Args section if available
        if [[ "$args" != "null" ]]; then
            echo "#### Args"
            echo ""
            echo "| Name | Description | Type | Required | Default | Docs |"
            echo "|------|-------------|------|----------|---------|------|"
            echo "$args" | jq -r '.[] |
                "| " +
                (if .Name and .Name != "" then .Name else "N/A" end) +
                " | " +
                (.Description | gsub("<"; "\\<") | gsub(">"; "\\>")) +
                " | " +
                (if .Type then .Type else "N/A" end) +
                " | " +
                (if .Required then "Yes" else "No" end) +
                " | " +
                (if .Default and .Default != "" then .Default else "N/A" end) +
                " | " +
                (if .Docs and .Docs != "" then .Docs else "N/A" end) +
                " |"'
            echo ""
        fi

        # Add Payloads section only if payloads exist
        if [[ "$payloads" != "null" ]]; then
            echo "#### Payloads"
            echo ""
            echo "**Type:** $payloads_type | **Required:** $payloads_required"
            echo ""
            echo "| Description | Docs |"
            echo "|-------------|------|"
            echo "$payloads" | jq -r '.[] |
                "| " +
                (.Description | gsub("<"; "\\<") | gsub(">"; "\\>")) +
                " | " +
                (if .Docs and .Docs != "" then .Docs else "N/A" end) +
                " |"'
            echo ""
        fi
    } >> "$markdown_file"
done

echo "Markdown documentation generated at $markdown_file"