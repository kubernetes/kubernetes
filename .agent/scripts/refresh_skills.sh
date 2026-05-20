#!/bin/bash

# Copyright The Kubernetes Authors.
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

# Ensure the script fails on error
set -e

SKILLS_DIR=".agent/skills"
SCRIPTS_DIR=".agent/scripts"
SOURCE_DIR="/google/src/head/depot/google3/learning/gemini/agents/skills"
INDEX_FILE=".agent/skills_index.md"

# Create directories if they don't exist
mkdir -p "$SKILLS_DIR"
mkdir -p "$SCRIPTS_DIR"

echo "Refresing skills from $SOURCE_DIR to $SKILLS_DIR..."

# Run Python to handle symlinking and complex frontmatter parsing
python3 - <<EOF
import os
import glob
import re

skills_dir = "$SKILLS_DIR"
source_dir = "$SOURCE_DIR"
index_file = "$INDEX_FILE"

# Clear existing symlinks in skills dir
for f in glob.glob(os.path.join(skills_dir, "*")):
    if os.path.islink(f):
        os.unlink(f)

# Get all skills (look for SKILL.md in subdirectories)
skills = glob.glob(os.path.join(source_dir, "*/SKILL.md"))

index_content = "# Available Skills\n\n"

print(f"Found {len(skills)} skills.")

for skill in sorted(skills):
    dir_name = os.path.basename(os.path.dirname(skill))
    symlink_path = os.path.join(skills_dir, f"{dir_name}.md")
    
    # Create symlink
    try:
        os.symlink(skill, symlink_path)
    except OSError as e:
        print(f"Error symlinking {dir_name}: {e}")
        continue
        
    # Read frontmatter
    try:
        with open(skill, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {dir_name}: {e}")
        continue
        
    # Match frontmatter
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if match:
        frontmatter = match.group(1)
        # Extract title and description
        title_match = re.search(r"^title:\s*(.*)$", frontmatter, re.MULTILINE)
        # Robust regex for description block (handles |, >, >-, |+ etc.)
        desc_match = re.search(r"^description:\s*(?:[|>][+-]?)?\s*\n((?:\s+.*(?:\n|$))*)", frontmatter, re.MULTILINE)
        
        title = title_match.group(1).strip() if title_match else dir_name
        desc = ""
        if desc_match:
            # De-indent the description
            lines = desc_match.group(1).split('\n')
            # Filter out empty lines for indentation check
            non_empty_lines = [l for l in lines if l.strip()]
            if non_empty_lines:
                min_indent = min(len(re.match(r"^\s*", l).group(0)) for l in non_empty_lines)
                desc = "\n".join(l[min_indent:] for l in lines).strip()
            
        index_content += f"## [{title}](file://{skill})\n{desc}\n\n"
    else:
        index_content += f"## [{dir_name}](file://{skill})\nNo description available.\n\n"

try:
    with open(index_file, 'w') as f:
        f.write(index_content)
    print(f"Generated {index_file}")
except Exception as e:
    print(f"Error writing index file: {e}")

EOF

echo "Done!"
