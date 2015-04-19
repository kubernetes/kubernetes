// Copyright 2011 Software Freedom Conservancy. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Utilities for working with WebDriver response objects.
 * @see: http://code.google.com/p/selenium/wiki/JsonWireProtocol#Responses
 */

goog.provide('bot.response');
goog.provide('bot.response.ResponseObject');

goog.require('bot.Error');
goog.require('bot.ErrorCode');


/**
 * Type definition for a response object, as defined by the JSON wire protocol.
 * @typedef {{status: bot.ErrorCode, value: (*|{message: string})}}
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol#Responses
 */
bot.response.ResponseObject;


/**
 * @param {*} value The value to test.
 * @return {boolean} Whether the given value is a response object.
 */
bot.response.isResponseObject = function(value) {
  return goog.isObject(value) && goog.isNumber(value['status']);
};


/**
 * Creates a new success response object with the provided value.
 * @param {*} value The response value.
 * @return {!bot.response.ResponseObject} The new response object.
 */
bot.response.createResponse = function(value) {
  if (bot.response.isResponseObject(value)) {
    return /** @type {!bot.response.ResponseObject} */ (value);
  }
  return {
    'status': bot.ErrorCode.SUCCESS,
    'value': value
  };
};


/**
 * Converts an error value into its JSON representation as defined by the
 * WebDriver wire protocol.
 * @param {(bot.Error|Error|*)} error The error value to convert.
 * @return {!bot.response.ResponseObject} The new response object.
 */
bot.response.createErrorResponse = function(error) {
  if (bot.response.isResponseObject(error)) {
    return /** @type {!bot.response.ResponseObject} */ (error);
  }

  var statusCode = error && goog.isNumber(error.code) ? error.code :
      bot.ErrorCode.UNKNOWN_ERROR;
  return {
    'status': /** @type {bot.ErrorCode} */ (statusCode),
    'value': {
      'message': (error && error.message || error) + ''
    }
  };
};


/**
 * Checks that a response object does not specify an error as defined by the
 * WebDriver wire protocol. If the response object defines an error, it will
 * be thrown. Otherwise, the response will be returned as is.
 * @param {!bot.response.ResponseObject} responseObj The response object to
 *     check.
 * @return {!bot.response.ResponseObject} The checked response object.
 * @throws {bot.Error} If the response describes an error.
 * @see http://code.google.com/p/selenium/wiki/JsonWireProtocol#Failed_Commands
 */
bot.response.checkResponse = function(responseObj) {
  var status = responseObj['status'];
  if (status == bot.ErrorCode.SUCCESS) {
    return responseObj;
  }

  // If status is not defined, assume an unknown error.
  status = status || bot.ErrorCode.UNKNOWN_ERROR;

  var value = responseObj['value'];
  if (!value || !goog.isObject(value)) {
    throw new bot.Error(status, value + '');
  }

  throw new bot.Error(status, value['message'] + '');
};
