[![Build Status](https://travis-ci.org/jasmine/jasmine-npm.png?branch=master)](https://travis-ci.org/jasmine/jasmine-npm)

# The Jasmine Module

The [Jasmine](https://github.com/pivotal/jasmine) module is a package of helper code for developing Jasmine projects for Node.js.

## Contents

This module allows you to run Jasmine specs for your Node.js code. The output will be displayed in your terminal.

## Installation

`npm install -g jasmine`

## Initializing

To initialize a project for Jasmine

`jasmine init`

To seed your project with some examples

`jasmine examples`

## Usage

To run your test suite

`jasmine`

## Configuration

Customize `spec/support/jasmine.json` to enumerate the source and spec files you would like the Jasmine runner to include.
You may use dir glob strings.

Alternatively, you may specify the path to your `jasmine.json` by setting an environment variable:

`jasmine JASMINE_CONFIG_PATH=relative/path/to/your/jasmine.json`

## Support

Jasmine Mailing list: [jasmine-js@googlegroups.com](mailto:jasmine-js@googlegroups.com)
Twitter: [@jasminebdd](http://twitter.com/jasminebdd)

Please file issues here at Github

Copyright (c) 2008-2013 Pivotal Labs. This software is licensed under the MIT License.
