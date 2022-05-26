# go-header
[![ci](https://github.com/denis-tingaikin/go-header/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/denis-tingaikin/go-header/actions/workflows/ci.yml)

Go source code linter providing checks for license headers.

## Installation

For installation you can simply use `go get`.

```bash
go get github.com/denis-tingaikin/go-header/cmd/go-header
```

## Configuration

To configuring `.go-header.yml` linter you simply need to fill the next fields:

```yaml
---
template: # expects header template string.
template-path: # expects path to file with license header string. 
values: # expects `const` or `regexp` node with values where values is a map string to string.
  const:
    key1: value1 # const value just checks equality. Note `key1` should be used in template string as {{ key1 }} or {{ KEY1 }}.
  regexp:
    key2: value2 # regexp value just checks regex match. The value should be a valid regexp pattern. Note `key2` should be used in template string as {{ key2 }} or {{ KEY2 }}.
```

Where `values` also can be used recursively. Example:

```yaml
values:
  const:
    key1: "value" 
  regexp:
    key2: "{{key1}} value1" # Reads as regex pattern "value value1"
```

## Bult-in values

- **YEAR** - Expects current year. Example header value: `2020`.  Example of template using: `{{YEAR}}` or `{{year}}`.
- **YEAR-RANGE** - Expects any valid year interval or current year. Example header value: `2020` or `2000-2020`. Example of template using: `{{year-range}}` or `{{YEAR-RANGE}}`.

## Execution

`go-header` linter expects file paths on input. If you want to run `go-header` only on diff files, then you can use this command:

```bash
go-header $(git diff --name-only | grep -E '.*\.go')
```

## Setup example

### Step 1

Create configuration file  `.go-header.yml` in the root of project.

```yaml
---
values:
  const:
    MY COMPANY: mycompany.com
template: |
  {{ MY COMPANY }}
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at:

  	  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
```

### Step 2 
You are ready! Execute `go-header ${PATH_TO_FILES}` from the root of the project. 
