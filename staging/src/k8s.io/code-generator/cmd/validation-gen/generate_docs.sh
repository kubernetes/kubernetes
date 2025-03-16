#!/bin/bash
# Copyright 2024 The Kubernetes Authors.
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

# Directory where the script is located (staging/src/k8s.io/code-generator/validation-gen/)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run the command and capture the JSON output
json_output=$(go run "${script_dir}" --docs)

# Create a Markdown file in the same directory
markdown_file="${script_dir}/validation_tags.md"

# Write the header to the Markdown file
{
    echo "# Kubernetes Validation Tags Documentation"
    echo ""
    echo "This document lists the supported validation tags and their related information."
    echo ""
    echo "## Tags Overview"
    echo ""
    echo "| Tag | Usage | Args | Description | Scopes |"
    echo "|-----|-------------|------|-------------|----------|"
} > "$markdown_file"

# Process the JSON output and populate the main table
{
    echo "$json_output" | jq -c '.[]' | while read -r line; do
        tag=$(echo "$line" | jq -r '.Tag')
        description=$(echo "$line" | jq -r '.Description')
        scopes=$(echo "$line" | jq -r '.Scopes | join(", ")')
        usage=$(echo "$line" | jq -r '.Usage')
        payloads=$(echo "$line" | jq -r '.Payloads')
        args=$(echo "$line" | jq -r '.Args')

        # Format Args for the main table
        if [[ "$args" != "null" ]]; then
            args_formatted=$(echo "$args" | jq -r '.[] | .Description' | paste -sd "," -)
        else
            args_formatted="N/A"
        fi

        # Add row to main table with link to payloads if they exist
        tag_link=$(echo "$tag" | sed 's/k8s:/k8s/' | tr '[:upper:]' '[:lower:]')
        # echo "| [\`$tag\`](#$tag_link) | $usage | $args_formatted | $description | $scopes |"
        # Properly format the usage string to include the field name.
        formatted_usage=$(echo "$usage" | sed 's/</\\</g' | sed 's/>/\\>/g')
        echo "| [\`$tag\`](#$tag_link) | $formatted_usage | $args_formatted | $description | $scopes |"
    done

    echo ""
    echo "## Tag Details"
    echo ""
} >> "$markdown_file"

# Create separate sections for each tag's payloads
echo "$json_output" | jq -c '.[]' | while read -r line; do
    tag=$(echo "$line" | jq -r '.Tag')
    payloads=$(echo "$line" | jq -r '.Payloads')
    args=$(echo "$line" | jq -r '.Args') # Keep this in case we need it later

    # Create section for this tag
    {
        echo "### $tag"
        echo ""

        if [[ "$payloads" != "null" ]]; then
            echo "#### Payloads"
            echo ""
            echo "| Description | Docs | Schema |"
            echo "|-------------|------|---------|"
            echo "$payloads" | jq -r '.[] |
                "| **" +
                (.Description | gsub("<"; "\\<") | gsub(">"; "\\>")) +
                "** | " +
                (if .Docs then .Docs else "" end) +
                " | " +
                (if .Schema then
                    (.Schema | map(
                        "- `" + .Key + "`: `" + .Value + "`" +
                        (if .Docs then " (" + .Docs + ")" else "" end) +
                        (if .Default then " (default: `" + .Default + "`)" else "" end)

                    ) | join("<br>"))
                else "None" end) +
                " |"'
            echo ""
        else
            echo "#### Payloads"
            echo ""
            echo "null"
            echo ""
        fi
    } >> "$markdown_file"
done

echo "Markdown documentation generated at $markdown_file"
